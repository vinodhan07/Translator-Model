import torch
import sentencepiece as spm
import pandas as pd
from train import Encoder, Decoder, Seq2Seq, device

BASE_DIR = r"D:\Project\Translator Model\translator_seq2seq"
TOKENIZER_PATH = f"{BASE_DIR}\\data\\tokenizers\\spm.model"
MODEL_PATH = f"{BASE_DIR}\\models\\translator_seq2seq.pt"

def translate_sentence(sentence):
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_PATH)
    vocab_size = sp.get_piece_size()

    enc = Encoder(vocab_size, 256, 512)
    dec = Decoder(vocab_size, 256, 512)
    model = Seq2Seq(enc, dec).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    tokens = [1] + sp.encode(sentence, out_type=int) + [2]
    src = torch.tensor(tokens).unsqueeze(0).to(device)
    hidden, cell = model.encoder(src)

    tgt_tokens = [1]
    for _ in range(30):
        tgt_tensor = torch.tensor([tgt_tokens[-1]]).to(device)
        output, hidden, cell = model.decoder(tgt_tensor, hidden, cell)
        pred_token = output.argmax(1).item()
        if pred_token == 2:
            break
        tgt_tokens.append(pred_token)

    translated_text = sp.decode(tgt_tokens[1:])
    print("🗣️ English:", sentence)
    print("🇫🇷 French:", translated_text)


if __name__ == "__main__":
    translate_sentence("how are you today?")
