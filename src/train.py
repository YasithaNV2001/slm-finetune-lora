import os
import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

def load_dataset(path="data/tokenized"):
    print(f"📂 Loading tokenized dataset from {path}")
    return load_from_disk(path)

def prepare_model(model_name="distilbert-base-uncased"):
    print(f"🔧 Loading base model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_lin", "v_lin"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    print("📦 Applying LoRA configuration...")
    model = get_peft_model(model, config)
    return model

def train():
    dataset = load_dataset()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = prepare_model()

    args = TrainingArguments(
        output_dir="outputs/checkpoints",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="outputs/logs",
        logging_steps=10,
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )

    print("🚀 Starting training...")
    trainer.train()
    print("✅ Training complete!")

    print("💾 Saving model to outputs/checkpoints")
    trainer.save_model("outputs/checkpoints")

if __name__ == "__main__":
    train()
