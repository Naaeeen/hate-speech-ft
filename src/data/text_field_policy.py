"""Shared HateXplain text-field construction policy.

Every method starts from the same text: join HateXplain `post_tokens` with one
space. We deliberately avoid dataset-level cleaning here, because extra
cleaning would make methods less comparable and could break alignment with the
original HateXplain rationale tokens. Tokenization/vectorization happens later,
inside each method.
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
Use `build_text_from_post_tokens(example)` when a method needs raw text. This is
equivalent to `" ".join(example["post_tokens"])`.

Transformer methods should call `tokenize_hatexplain_text(...)`, which builds
that shared text and then calls the model tokenizer with truncation/max length.
TF-IDF and BiLSTM should call `build_text_from_post_tokens(...)` and pass the
same string into their own vectorizer/tokenizer.

Do not add punctuation removal, stopword removal, stemming, profanity masking,
metadata concatenation, or other dataset-level cleaning here. Those choices
would change the shared input before the model-specific code sees it.
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
