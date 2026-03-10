import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "../models/weibo_sentiment_bert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

def predict(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = torch.argmax(probs).item()

    label = model.config.id2label[pred_id]
    return {
        "text": text,
        "pred_id": pred_id,
        "label": label,
        "prob_negative": round(probs[0].item(), 4),
        "prob_positive": round(probs[1].item(), 4),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('用法: python scripts/predict_sentiment.py "这家店服务好，下次还来"')
        sys.exit(1)

    text = sys.argv[1]
    result = predict(text)
    print(result)