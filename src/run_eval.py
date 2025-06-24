from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("outputs/checkpoints")
tokenizer = AutoTokenizer.from_pretrained("outputs/checkpoints")


# Load dataset (SST2)
dataset = load_dataset("glue", "sst2", split="validation[:200]")

# Load evaluation metric
accuracy_metric = evaluate.load("accuracy")

# Run evaluation
model.eval()
for item in dataset:
    inputs = tokenizer(item["sentence"], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        accuracy_metric.add(prediction=prediction, reference=item["label"])

# Print result
result = accuracy_metric.compute()
print(f"✅ Accuracy on SST2 validation set: {result['accuracy']:.4f}")
