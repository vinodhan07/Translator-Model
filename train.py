import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---------------- CONFIG ----------------
BASE_DIR = r"D:\Project\Translator Model\translator_seq2seq"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
TOKENIZER_PATH = os.path.join(BASE_DIR, "data", "tokenizers", "spm.model")
MODEL_PATH = os.path.join(BASE_DIR, "models", "translator_seq2seq.pt")

BATCH_SIZE = 32
EMB_DIM = 256
HID_DIM = 512
N_EPOCHS = 5
LR = 1e-3
MAX_TGT_LEN = 100  # safety cap
CLIP = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- Dataset ---------------- #
class TranslationDataset(Dataset):
    def __init__(self, csv_file, sp_processor, max_len=200):
        self.df = pd.read_csv(csv_file)
        self.sp = sp_processor
        self.bos = self.sp.bos_id() if hasattr(self.sp, "bos_id") else 1
        self.eos = self.sp.eos_id() if hasattr(self.sp, "eos_id") else 2
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def encode_sentence(self, text):
        ids = self.sp.encode(text, out_type=int)
        # truncate if too long (leave room for bos/eos)
        if len(ids) > self.max_len - 2:
            ids = ids[: (self.max_len - 2)]
        return [self.bos] + ids + [self.eos]

    def __getitem__(self, idx):
        src_text = str(self.df.iloc[idx]["english"])
        tgt_text = str(self.df.iloc[idx]["french"])
        src_ids = torch.tensor(self.encode_sentence(src_text), dtype=torch.long)
        tgt_ids = torch.tensor(self.encode_sentence(tgt_text), dtype=torch.long)
        return src_ids, tgt_ids


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded, tgt_padded


# ---------------- Model ---------------- #
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, hidden, cell):
        # input: (batch,) containing token ids (not one-hot)
        input = input.unsqueeze(1)  # (batch, 1)
        embedded = self.embedding(input)  # (batch,1,emb)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))  # (batch, vocab)
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        # src: (batch, src_len), tgt: (batch, tgt_len)
        batch_size, tgt_len = tgt.shape
        vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=src.device)
        hidden, cell = self.encoder(src)
        input_tok = tgt[:, 0]  # first token (<bos>)

        for t in range(1, tgt_len):
            output, hidden, cell = self.decoder(input_tok, hidden, cell)
            outputs[:, t] = output
            top1 = output.argmax(1)
            use_teacher = (torch.rand(1).item() < teacher_forcing_ratio)
            input_tok = tgt[:, t] if use_teacher else top1

        return outputs


# ---------------- Helpers ---------------- #
def ensure_sentencepiece(tokenizer_path):
    """
    Ensure a trained SentencePiece model exists. If not, try to import and call utils.data_prep.train_sentencepiece(...)
    """
    if os.path.exists(tokenizer_path):
        print("Found tokenizer:", tokenizer_path)
        return

    print("Tokenizer not found at:", tokenizer_path)
    print("Attempting to train tokenizer by calling utils.data_prep.train_sentencepiece(...)")

    # Add base dir to sys.path so utils can be imported when running from project root
    project_root = os.path.dirname(BASE_DIR)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        import utils.data_prep as dp
    except Exception as e:
        print("Failed to import utils.data_prep:", e)
        raise RuntimeError(
            "Cannot find utils.data_prep to train tokenizer automatically. "
            "Please run `python utils/data_prep.py` manually."
        )

    # load train.csv for sampling
    train_csv = os.path.join(DATA_DIR, "train.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing processed train.csv at {train_csv}. Run data_prep.py first.")

    train_df = pd.read_csv(train_csv)
    # call train_sentencepiece from utils.data_prep; dp.train_sentencepiece takes train_df as arg
    try:
        # use a sampled corpus to speed up tokeniser training if df is large
        dp.train_sentencepiece(train_df, vocab_size=8000, max_corpus_lines=200000)
    except TypeError:
        # Older version of the helper may not accept kwargs
        dp.train_sentencepiece(train_df)
    print("Tokenizer training completed (if no errors).")


def evaluate(model, dataloader, criterion, vocab_size):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            output = model(src, tgt, teacher_forcing_ratio=0.0)  # no teacher forcing for eval
            output = output[:, 1:].reshape(-1, vocab_size)
            tgt_flat = tgt[:, 1:].reshape(-1)
            loss = criterion(output, tgt_flat)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)


# ---------------- Train ---------------- #
def train_model():
    # ensure tokenizer exists (calls utils.data_prep if needed)
    ensure_sentencepiece(TOKENIZER_PATH)

    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)
    vocab_size = sp.get_piece_size()
    print("SentencePiece vocab size:", vocab_size)

    # datasets & dataloaders
    train_csv = os.path.join(DATA_DIR, "train.csv")
    val_csv = os.path.join(DATA_DIR, "val.csv")
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train CSV not found at {train_csv}. Run data_prep.py first or check path.")

    train_ds = TranslationDataset(train_csv, sp, max_len=200)
    val_ds = TranslationDataset(val_csv, sp, max_len=200) if os.path.exists(val_csv) else None

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn) if val_ds else None

    # model
    encoder = Encoder(vocab_size, EMB_DIM, HID_DIM)
    decoder = Decoder(vocab_size, EMB_DIM, HID_DIM)
    model = Seq2Seq(encoder, decoder).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("🚀 Training started...")
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{N_EPOCHS}")
        for src, tgt in pbar:
            src = src.to(device)
            tgt = tgt.to(device)
            # randomize teacher forcing ratio per batch (you can keep fixed)
            teacher_forcing_ratio = 0.5

            output = model(src, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
            output_dim = output.shape[-1]
            output_flat = output[:, 1:].reshape(-1, output_dim)
            tgt_flat = tgt[:, 1:].reshape(-1)

            loss = criterion(output_flat, tgt_flat)

            optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=epoch_loss / (pbar.n + 1))

        avg_train_loss = epoch_loss / len(train_dl)
        print(f"\nEpoch {epoch+1} train loss: {avg_train_loss:.4f}")

        # run a quick validation pass
        if val_dl is not None:
            val_loss = evaluate(model, val_dl, criterion, vocab_size)
            print(f"Epoch {epoch+1} val loss: {val_loss:.4f}")

        # save intermediate checkpoint
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        ckpt_path = MODEL_PATH.replace(".pt", f".epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    # final save
    torch.save(model.state_dict(), MODEL_PATH)
    print("✅ Final model saved to", MODEL_PATH)


if __name__ == "__main__":
    train_model()