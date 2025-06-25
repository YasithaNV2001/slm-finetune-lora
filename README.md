# 🔬 SLM Fine-tuning with LoRA (Gradio UI)

This project demonstrates how to fine-tune a **Small Language Model (SLM)** using **LoRA (Low-Rank Adaptation)** and deploy it with a simple **Gradio-based UI**. The workflow uses Hugging Face 🤗 Transformers and PEFT libraries.

---

## 📦 Project Structure
```bash
slm-finetune-lora/
│
├── app.py # Gradio UI app for inference
├── test_app.py # Simple script to test predictions
├── train.py # Fine-tuning script using LoRA
├── model/ # Folder where the trained model gets saved
├── data/
│ └── sentiment.csv # Dataset used for fine-tuning
├── requirements.txt # Dependencies
├── README.md # This file
└── .gitignore

```
---

## 🔧 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/YasithaNV/slm-finetune-lora.git
cd slm-finetune-lora
```
### 2.Create & Activate Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate       # For Linux/macOS
.venv\Scripts\activate          # For Windows
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

📁 Dataset Format (CSV)

data/sentiment.csv should look like this:

```csv

text,label
I love this product!,1
This is terrible,0
It was okay, not bad,1
```
1 = Positive
0 = Negative

## 🧠 Fine-Tune Model with LoRA
Run this script to fine-tune the model:
```bash
python train.py
```
This saves the fine-tuned model under the model/ directory.

## 🖥️ Run Gradio UI (App)
```bash
python app.py
```
Opens: http://127.0.0.1:7860
Type your text and get model predictions.
To create a public link, you can change:
```bash
gr.Interface(...).launch(share=True)
```

## ✅ Test from CLI (Without UI)
```bash
python test_app.py
```
You will see predictions in terminal like:

Input: I love this product!
Prediction: Positive 😊




