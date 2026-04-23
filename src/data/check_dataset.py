from datasets import load_dataset

def show_split_info(ds):
    for split in ds.keys():
        print(f"{split}: {len(ds[split])}")

def main():
    ds = load_dataset("Hate-speech-CNERG/hatexplain")
    show_split_info(ds)

    sample = ds["train"][0]
    print("\nKeys:", sample.keys())
    print("\nFirst 20 tokens:", sample["post_tokens"][:20])

if __name__ == "__main__":
    main()