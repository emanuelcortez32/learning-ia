import pandas as pd
import torch

from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
from pathlib import Path

DEVICE = "cpu"
BATCH_SIZE = 32
MODEL_NAME = "Helsinki-NLP/opus-mt-en-es"

current_dir = Path(__file__).resolve()
current_dir_parent = current_dir.parent
top_folder = current_dir_parent.parent
out_folder = top_folder / "out"

tokenizer = MarianTokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME).to(DEVICE)

def translate_batch(texts):
    tokens = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_lenght=512
    ).to(DEVICE)

    with torch.no_grad():
        translated = model.generate(**tokens)

    return tokenizer.batch_decode(translated, skip_special_tokens=True)


def translate_column(texts):
    translated_texts = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[i:i+BATCH_SIZE]
        translated_texts.extend(translate_batch(batch))
    
    return translated_texts

def process_file(name, input_path):

    df = pd.read_parquet("hf://datasets/JasperLS/prompt-injections/" + input_path)

    assert "text" in df.columns
    assert "label" in df.columns

    texts = df["text"].astype(str).tolist()
    labels = df["label"]

    translated_texts = translate_column(texts)

    translated_df = pd.DataFrame({
        "text": translated_texts,
        "label": labels
    })

    print(translated_df.head())
    print("\n--------")
    print(translated_df.info())
    
    out_path = f"{out_folder}/{name}.parquet"

    translated_df.to_parquet(out_path, index=False)

splits = {'train': 'data/train-00000-of-00001-9564e8b05b4757ab.parquet', 'test': 'data/test-00000-of-00001-701d16158af87368.parquet'}

process_file("train_es", splits["train"])
process_file("test_es", splits["test"])


