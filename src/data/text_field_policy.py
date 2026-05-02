"""Shared HateXplain text-field construction policy.
We do not clean the HateXplain text because post_tokens is already
the canonical annotated text representation. We only join tokens with spaces
to reconstruct a text string. Any further tokenization should be method-specific,
not dataset-level preprocessing, so that all methods are compared on the same input content.

Team policy:
All experiment methods must construct the model input from HateXplain
`post_tokens` by joining tokens with a single space. No extra dataset-level
cleaning is applied before model-specific tokenization or vectorization.

This keeps every method comparable and preserves the token sequence that
HateXplain rationales are aligned to.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


TEXT_FIELD_POLICY = (
    "Use HateXplain post_tokens as the single source of text input. "
    "Construct text with: text = ' '.join(example['post_tokens']). "
    "No extra dataset-level cleaning, stemming, lemmatization, stopword "
    "removal, punctuation removal, emoji removal, profanity masking, or "
    "metadata concatenation should be applied before each method's own "
    "tokenizer or vectorizer."
)

TEXT_FIELD_USAGE = """
How to use this module
======================

What is `example`?
------------------
`example` means one row/sample from the HateXplain dataset. When using the
Hugging Face datasets library, a typical example is:

    from datasets import load_dataset

    ds = load_dataset("Hate-speech-CNERG/hatexplain")
    example = ds["train"][0]

That `example` is a dictionary-like object containing the fields declared by
the Hugging Face dataset schema:

    {
        "id": "...",
        "post_tokens": ["this", "is", "a", "post"],
        "annotators": ...,
        "rationales": [...]
    }

The helper functions in this module only require `post_tokens`. The example
above is therefore a minimal shape for text construction, not a full printout
of every nested annotation field.

For reference, the raw HateXplain JSON used by the Hugging Face loader stores
`annotators` as a list of annotator dictionaries:

    {
        "post_id": "...",
        "post_tokens": ["this", "is", "a", "post"],
        "annotators": [
            {"label": "hatespeech", "annotator_id": 203, "target": ["..."]},
            {"label": "offensive", "annotator_id": 204, "target": ["..."]},
            {"label": "normal", "annotator_id": 233, "target": ["..."]}
        ],
        "rationales": [[0, 1, 0, 0], [0, 0, 0, 0]]
    }

In the Hugging Face dataset schema, `annotators.label` is declared as a
ClassLabel with names `hatespeech`, `normal`, and `offensive`. Depending on the
library representation being printed, labels may appear as class ids rather
than strings. This does not affect text construction because text construction
uses only `post_tokens`.

Shared dataset-level text construction
--------------------------------------
Every method should construct the input text through this helper:

    from src.data.text_field_policy import build_text_from_post_tokens

    text = build_text_from_post_tokens(example)

This is equivalent to:

    text = " ".join(example["post_tokens"])

Do not add any extra dataset-level cleaning before this text enters a
method-specific tokenizer or vectorizer. In particular, do not remove
punctuation, emojis, stopwords, hashtags, offensive terms, or apply stemming,
lemmatization, spelling correction, profanity masking, or metadata concatenation.

Transformer methods
-------------------
For DistilBERT, BERT, RoBERTa, LoRA, frozen-backbone, partial fine-tuning,
full fine-tuning, and LP-FT experiments, first build the shared text and then
call the model tokenizer:

    from src.data.text_field_policy import tokenize_hatexplain_text

    tokenized = tokenize_hatexplain_text(
        example,
        tokenizer,
        max_length=128,
    )

This calls:

    tokenizer(text, truncation=True, max_length=128)

Classical baselines
-------------------
For TF-IDF and Logistic Regression, use the same shared text as input to the
vectorizer:

    text = build_text_from_post_tokens(example)
    features = tfidf_vectorizer.transform([text])

If `TfidfVectorizer` lowercases internally or creates n-grams, that is part of
the vectorizer configuration. It is not a separate dataset-level text policy.

Bi-LSTM baseline
----------------
For a Bi-LSTM trained from scratch, also start from the same shared text:

    text = build_text_from_post_tokens(example)

Then pass that text into the Bi-LSTM vocabulary/tokenization pipeline chosen for
that baseline. Document the Bi-LSTM tokenizer separately, but do not mutate the
shared HateXplain text field.

Why this policy exists
----------------------
HateXplain provides `post_tokens` as the canonical annotated token sequence, and
its human rationale masks are aligned to that sequence by index. Extra cleaning
can shift, delete, or rewrite tokens, making rationale alignment harder to
interpret. This project compares model adaptation methods, so all methods should
start from the same textual content.
""".strip()


class TextTokenizer(Protocol):
    """Minimal tokenizer interface used by Hugging Face tokenizers."""

    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]:
        ...


def build_text_from_post_tokens(example: Mapping[str, Any]) -> str:
    """Construct the shared HateXplain text field.

    Args:
        example: One HateXplain row/sample, such as `ds["train"][0]` after
            calling `load_dataset("Hate-speech-CNERG/hatexplain")`.

    Returns:
        A string built by joining `example["post_tokens"]` with single spaces.

    The dataset provides `post_tokens` as the annotated token sequence. We only
    join those tokens with single spaces. Do not remove punctuation, emojis,
    hashtags, stopwords, or other tokens here; those choices would change the
    shared input and may break rationale alignment.
    """

    if "post_tokens" not in example:
        raise KeyError("HateXplain example is missing required field 'post_tokens'.")

    post_tokens = example["post_tokens"]
    if isinstance(post_tokens, str) or not isinstance(post_tokens, Sequence):
        raise TypeError("'post_tokens' must be a sequence of tokens, not raw text.")

    return " ".join(str(token) for token in post_tokens)


def tokenize_hatexplain_text(
    example: Mapping[str, Any],
    tokenizer: TextTokenizer,
    max_length: int,
    **tokenizer_kwargs: Any,
) -> Mapping[str, Any]:
    """Build the shared text field, then apply a model-specific tokenizer.

    Args:
        example: One HateXplain row/sample containing `post_tokens`.
        tokenizer: A Hugging Face-style tokenizer or compatible callable.
        max_length: Maximum encoded sequence length for truncation.
        **tokenizer_kwargs: Extra tokenizer options. These override the default
            `truncation=True` and `max_length=max_length` options if repeated.

    Returns:
        The tokenizer output, typically a mapping containing `input_ids` and
        `attention_mask`.

    Transformer methods should use this path for sequence classification.
    Classical baselines should call `build_text_from_post_tokens` and pass the
    returned string into their vectorizer instead.
    """

    text = build_text_from_post_tokens(example)
    options = {"truncation": True, "max_length": max_length}
    options.update(tokenizer_kwargs)
    return tokenizer(text, **options)
