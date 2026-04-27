"""
Preprocessing module for financial tweet text cleaning and normalization.

Provides composable functions and named pipelines (pp_minimal, pp_aggressive,
pp_transformer) that can be A/B tested with the same downstream classifier.
"""

import re
import unicodedata
from typing import List, Optional, Callable

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure NLTK data is available
for resource in ["stopwords", "wordnet", "punkt", "punkt_tab",
                 "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.data.find(f"corpora/{resource}" if resource in ["stopwords", "wordnet"]
                       else f"taggers/{resource}" if "tagger" in resource
                       else f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

# Try to import emoji; fall back gracefully
try:
    import emoji
    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False

# ---------------------------------------------------------------------------
# Singletons (avoid re-creating on every call)
# ---------------------------------------------------------------------------
_tweet_tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True,
                                  strip_handles=True)
_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()

# Financial-aware stopword list: keep negations and directional words
_KEEP_WORDS = {
    "no", "not", "nor", "against", "up", "down", "over", "under",
    "more", "less", "few", "above", "below", "before", "after",
}
_stopwords = set(stopwords.words("english")) - _KEEP_WORDS


# ===========================================================================
# Individual cleaning functions (composable)
# ===========================================================================

def clean_urls(text: str) -> str:
    """Remove URLs (http/https/www)."""
    return re.sub(r"http\S+|www\.\S+", "", text)


def clean_mentions(text: str) -> str:
    """Remove @mentions."""
    return re.sub(r"@\w+", "", text)


def clean_hashtags(text: str, mode: str = "keep_word") -> str:
    """Handle hashtags.

    Args:
        mode: 'keep_word' — strip '#' but keep the word (default).
              'drop' — remove entire hashtag.
    """
    if mode == "drop":
        return re.sub(r"#\w+", "", text)
    return re.sub(r"#(\w+)", r"\1", text)


def clean_tickers(text: str) -> str:
    """Replace $TICKER cashtags with <TICKER> placeholder.

    Preserves the signal that a ticker was mentioned without overfitting
    to specific stock names.
    """
    return re.sub(r"\$[A-Za-z]{1,5}\b", "<TICKER>", text)


def clean_numbers(text: str) -> str:
    """Replace percentages with <PCT> and other numbers with <NUM>.

    Preserves the signal that numerical claims exist in the tweet.
    """
    text = re.sub(r"\d+\.?\d*\s*%", "<PCT>", text)
    text = re.sub(r"\b\d+[\d,\.]*\b", "<NUM>", text)
    return text


def clean_emojis(text: str) -> str:
    """Convert emojis to textual descriptions using emoji.demojize.

    Falls back to removing non-ASCII if emoji package is not installed.
    """
    if HAS_EMOJI:
        return emoji.demojize(text, delimiters=(" ", " "))
    # Fallback: remove common emoji Unicode ranges
    return re.sub(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        r"\U00002702-\U000027B0\U000024C2-\U0001F251]+",
        "", text
    )


def clean_html(text: str) -> str:
    """Remove HTML entities (&amp; &lt; etc.)."""
    import html
    return html.unescape(text)


def clean_repeated_punctuation(text: str) -> str:
    """Collapse repeated punctuation (e.g., '!!!' -> '!') and whitespace."""
    text = re.sub(r"([!?.]){2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_unicode(text: str) -> str:
    """Apply NFKC Unicode normalization."""
    return unicodedata.normalize("NFKC", text)


def lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


# ===========================================================================
# Tokenization
# ===========================================================================

def tokenize(text: str, method: str = "tweet") -> List[str]:
    """Tokenize text.

    Args:
        method: 'tweet' — NLTK TweetTokenizer (preserves emoticons).
                'whitespace' — simple whitespace split.
    """
    if method == "whitespace":
        return text.split()
    return _tweet_tokenizer.tokenize(text)


# ===========================================================================
# Token-level operations
# ===========================================================================

def remove_stopwords(tokens: List[str],
                     extra_keep: Optional[set] = None) -> List[str]:
    """Remove stopwords, keeping negations and financial directional words.

    The default stoplist has been pruned of words that flip sentiment
    (no, not, nor) and directional words (up, down, over, under, etc.).
    """
    keep = _KEEP_WORDS | (extra_keep or set())
    return [t for t in tokens if t not in _stopwords or t in keep]


def stem_tokens(tokens: List[str]) -> List[str]:
    """Apply Porter stemming."""
    return [_stemmer.stem(t) for t in tokens]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Apply WordNet lemmatization."""
    return [_lemmatizer.lemmatize(t) for t in tokens]


# ===========================================================================
# Named pipelines
# ===========================================================================

def pp_minimal(text: str, return_tokens: bool = False):
    """Minimal preprocessing — light cleaning suitable for most models.

    Steps: HTML unescape → URLs → mentions → hashtags (keep word) →
    tickers → emojis (demojize) → Unicode normalize → lowercase →
    collapse punctuation.
    """
    text = clean_html(text)
    text = clean_urls(text)
    text = clean_mentions(text)
    text = clean_hashtags(text, mode="keep_word")
    text = clean_tickers(text)
    text = clean_emojis(text)
    text = normalize_unicode(text)
    text = lowercase(text)
    text = clean_repeated_punctuation(text)
    if return_tokens:
        return tokenize(text, method="tweet")
    return text


def pp_aggressive(text: str, return_tokens: bool = False,
                  token_processing: str = "lemma"):
    """Aggressive preprocessing — full normalization pipeline.

    Steps: everything in pp_minimal + numbers → stopword removal →
    stemming OR lemmatization.

    Args:
        token_processing: 'stem' for Porter stemming, 'lemma' for
            WordNet lemmatization (default).
    """
    text = clean_html(text)
    text = clean_urls(text)
    text = clean_mentions(text)
    text = clean_hashtags(text, mode="keep_word")
    text = clean_tickers(text)
    text = clean_numbers(text)
    text = clean_emojis(text)
    text = normalize_unicode(text)
    text = lowercase(text)
    text = clean_repeated_punctuation(text)

    tokens = tokenize(text, method="tweet")
    tokens = remove_stopwords(tokens)

    if token_processing == "stem":
        tokens = stem_tokens(tokens)
    else:
        tokens = lemmatize_tokens(tokens)

    if return_tokens:
        return tokens
    return " ".join(tokens)


def pp_transformer(text: str) -> str:
    """Minimal cleaning for Transformer inputs.

    Pretrained tokenizers handle casing, subword splitting, etc.
    We only normalize URLs/mentions and convert emojis to text so
    the tokenizer gets cleaner input without losing semantic content.
    """
    text = clean_html(text)
    text = clean_urls(text)
    text = clean_mentions(text)
    text = clean_hashtags(text, mode="keep_word")
    text = clean_tickers(text)
    text = clean_emojis(text)
    text = clean_repeated_punctuation(text)
    return text


# ===========================================================================
# Batch processing utility
# ===========================================================================

def preprocess_corpus(texts, pipeline: Callable = pp_minimal, **kwargs):
    """Apply a preprocessing pipeline to a list/Series of texts.

    Returns a list of processed strings (or token lists if return_tokens=True).
    """
    return [pipeline(str(t), **kwargs) for t in texts]
