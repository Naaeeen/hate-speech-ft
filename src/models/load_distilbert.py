from transformers import AutoTokenizer, AutoModelForSequenceClassification

def main():
    model_name = "distilbert-base-uncased"
    num_labels = 3

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    print("Tokenizer loaded:", model_name)
    print("Model loaded:", model_name)
    print("Number of labels:", model.config.num_labels)
    print("Classifier:", model.classifier)

if __name__ == "__main__":
    main()