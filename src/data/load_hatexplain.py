from datasets import load_dataset

def main():
    ds = load_dataset(
        "Hate-speech-CNERG/hatexplain",
        trust_remote_code=True
    )

    print(ds)
    print("\nTrain sample:")
    print(ds["train"][0])

if __name__ == "__main__":
    main()