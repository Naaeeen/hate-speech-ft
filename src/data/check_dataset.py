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

# train: 15383
# validation: 1922
# test: 1924
#
# Keys: dict_keys(['id', 'annotators', 'rationales', 'post_tokens'])
#
# First 20 tokens: ['u', 'really', 'think', 'i', 'would', 'not', 'have', 'been', 'raped', 'by', 'feral', 'hindu', 'or', 'muslim', 'back', 'in', 'india', 'or', 'bangladesh', 'and']
