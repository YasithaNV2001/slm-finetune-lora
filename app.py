from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import gradio as gr

# ✅ Load your fine-tuned model from local checkpoint
tokenizer = AutoTokenizer.from_pretrained("outputs/checkpoints")
model = AutoModelForSequenceClassification.from_pretrained("outputs/checkpoints")

# ✅ Create pipeline for inference
text_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# ✅ Define prediction function
def predict_sentiment(text):
    result = text_classifier(text)[0]
    label = result["label"]
    score = result["score"]
    return f"Prediction: {label} (Confidence: {score:.2f})"

# ✅ Gradio UI
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Enter text here..."),
    outputs="text",
    title="Sentiment Classifier",
    description="Enter a sentence and get the predicted sentiment using your fine-tuned model."
)

# ✅ Run with public link
interface.launch(share=True)
