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

# DatasetDict({
#     train: Dataset({
#         features: ['id', 'annotators', 'rationales', 'post_tokens'],
#         num_rows: 15383
#     })
#     validation: Dataset({
#         features: ['id', 'annotators', 'rationales', 'post_tokens'],
#         num_rows: 1922
#     })
#     test: Dataset({
#         features: ['id', 'annotators', 'rationales', 'post_tokens'],
#         num_rows: 1924
#     })
# })
#
# Train sample:
# {'id': '23107796_gab', 'annotators': {'label': [0, 2, 2], 'annotator_id': [203, 204, 233], 'target': [['Hindu', 'Islam'], ['Hindu', 'Islam'], ['Hindu', 'Islam', 'Other']]}, 'rationales': [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], 'post_tokens': ['u', 'really', 'think', 'i', 'would', 'not', 'have', 'been', 'raped', 'by', 'feral', 'hindu', 'or', 'muslim', 'back', 'in', 'india', 'or', 'bangladesh', 'and', 'a', 'neo', 'nazi', 'would', 'rape', 'me', 'as', 'well', 'just', 'to', 'see', 'me', 'cry']}
