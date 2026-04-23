import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3
    )

    texts = [
        "This is a normal sentence.",
        "This post contains offensive content."
    ]

    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**batch)

    print("Input IDs shape:", batch["input_ids"].shape)
    print("Logits shape:", outputs.logits.shape)

if __name__ == "__main__":
    main()