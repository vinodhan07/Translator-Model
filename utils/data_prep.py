# utils/data_prep.py
import os
import pandas as pd
import sentencepiece as spm
import random

# CONFIG: adjust these paths if your project layout differs
RAW_FILE = r"D:\Project\Translator Model\data\raw\train-00000-of-00001.parquet"
PROCESSED_DIR = r"D:\Project\Translator Model\translator_seq2seq\data\processed"
TOKENIZER_DIR = r"D:\Project\Translator Model\translator_seq2seq\data\tokenizers"
SPM_MODEL = os.path.join(TOKENIZER_DIR, "spm.model")

# Create dirs if missing
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)


def load_dataset(path):
    print("📖 Loading Parquet file...")
    df = pd.read_parquet(path)

    # If there's a 'translation' column with dicts, expand it
    if "translation" in df.columns:
        try:
            expanded = pd.DataFrame(df["translation"].tolist())
            # If contains 'en'/'fr' prefer that expanded table
            if "en" in expanded.columns and "fr" in expanded.columns:
                df = expanded
        except Exception:
            pass

    # Normalize common column names to 'english' and 'french'
    rename_map = {}
    for c in df.columns:
        if not isinstance(c, str):
            continue
        lc = c.lower()
        if lc in ("en", "english", "src", "source"):
            rename_map[c] = "english"
        if lc in ("fr", "french", "tgt", "target"):
            rename_map[c] = "french"
    df = df.rename(columns=rename_map)

    # If still no english/french columns, try heuristics
    if "english" not in df.columns or "french" not in df.columns:
        text_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
        if len(text_cols) >= 2:
            avg_len = {c: df[c].dropna().astype(str).map(len).mean() for c in text_cols}
            sorted_cols = sorted(avg_len.items(), key=lambda x: x[1], reverse=True)
            df = df.rename(columns={sorted_cols[0][0]: "english", sorted_cols[1][0]: "french"})
        else:
            raise ValueError("Could not locate English/French columns in the parquet file. Inspect the file.")

    df = df[["english", "french"]].dropna().reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    print("✅ Loaded dataset:", len(df), "rows")
    return df


def clean_text(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.lower().strip().split())


def split_data(df, train_ratio=0.90, val_ratio=0.05):
    total = len(df)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    print(f"✅ Saved train ({len(train_df)}), val ({len(val_df)}), test ({len(test_df)})")
    return train_df, val_df, test_df


def train_sentencepiece(train_df, vocab_size=8000, max_corpus_lines=200000, model_type="bpe"):
    """
    Train SentencePiece safely on Windows:
      - write corpus file inside TOKENIZER_DIR
      - chdir into TOKENIZER_DIR and call trainer with relative filenames
      - this avoids Windows quoting/parsing issues with long paths that contain spaces
    """
    # Skip if model exists
    if os.path.exists(SPM_MODEL):
        print(f"ℹ️  Found existing SentencePiece model at {SPM_MODEL}. Skipping training.")
        return

    print("🔤 Preparing corpus for SentencePiece training...")

    en_lines = train_df["english"].astype(str).tolist()
    fr_lines = train_df["french"].astype(str).tolist()
    all_lines = en_lines + fr_lines

    if max_corpus_lines is not None and len(all_lines) > max_corpus_lines:
        random.seed(42)
        all_lines = random.sample(all_lines, max_corpus_lines)
        print(f"ℹ️  Sampled {len(all_lines)} lines for tokenizer training (max_corpus_lines={max_corpus_lines})")
    else:
        print(f"ℹ️  Using {len(all_lines)} total lines for tokenizer training")

    corpus_file = os.path.join(TOKENIZER_DIR, "spm_corpus.txt")
    with open(corpus_file, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line.strip() + "\n")

    orig_cwd = os.getcwd()
    try:
        # move to tokenizer dir to run trainer with relative paths (avoids quoting issues)
        os.chdir(TOKENIZER_DIR)
        spm_cmd = f'--input=spm_corpus.txt --model_prefix=spm --vocab_size={vocab_size} --character_coverage=1.0 --model_type={model_type}'
        print("🔁 Running SentencePieceTrainer (inside tokenizer dir) with command:")
        print(spm_cmd)
        spm.SentencePieceTrainer.Train(spm_cmd)
    except Exception as e:
        print("❌ SentencePiece training failed with exception:")
        print(e)
        print("→ Try upgrading sentencepiece: pip install --upgrade sentencepiece")
        print("→ Or move the project to a path without spaces, e.g. D:\\translator_project\\translator_seq2seq")
        raise
    finally:
        os.chdir(orig_cwd)
        # cleanup corpus file
        try:
            os.remove(corpus_file)
        except Exception:
            pass

    print("✅ Tokenizer trained and saved as spm.model / spm.vocab in", TOKENIZER_DIR)


if __name__ == "__main__":
    # 1) load
    df = load_dataset(RAW_FILE)

    # 2) clean
    df["english"] = df["english"].apply(clean_text)
    df["french"] = df["french"].apply(clean_text)

    # 3) split & write CSVs
    train_df, val_df, test_df = split_data(df)

    # 4) train sentencepiece
    # max_corpus_lines=200000 speeds training on big corpora; set to None to use everything
    train_sentencepiece(train_df, vocab_size=8000, max_corpus_lines=200000)
