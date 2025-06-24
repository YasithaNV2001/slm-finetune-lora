from datasets import load_dataset
from transformers import AutoTokenizer
import os

def preprocess_sst2(tokenizer_name="distilbert-base-uncased", max_length=128):
    dataset = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_function(example):
        return tokenizer(
            example["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized_datasets

if __name__ == "__main__":
    output_dir = "data/tokenized"
    os.makedirs(output_dir, exist_ok=True)

    print("🔁 Tokenizing SST2 dataset using DistilBERT...")
    tokenized_datasets = preprocess_sst2()
    tokenized_datasets.save_to_disk(output_dir)
    print(f"✅ Tokenized dataset saved to: {output_dir}")
