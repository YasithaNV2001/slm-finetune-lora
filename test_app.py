from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the tokenizer and model from the fine-tuned checkpoint
tokenizer = AutoTokenizer.from_pretrained("outputs/checkpoints")
model = AutoModelForSequenceClassification.from_pretrained("outputs/checkpoints")

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return "Positive 😊" if prediction == 1 else "Negative 😠"

# Simple test cases
test_inputs = [
    "I love this product!",
    "This is terrible.",
    "It was okay, not bad."
]

for text in test_inputs:
    pred = predict_sentiment(text)
    print(f"Input: {text}")
    print(f"Prediction: {pred}")
    print("✅" if (
        (text == "I love this product!" and pred == "Positive 😊") or
        (text == "This is terrible." and pred == "Negative 😠") or
        (text == "It was okay, not bad." and pred == "Positive 😊")
    ) else "❌")
