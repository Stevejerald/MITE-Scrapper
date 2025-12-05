"""
preprocess.py

Normalization, tokenization and lemmatization utilities used by the matching engine.

Save as:
product_matcher/backend/app/matching/preprocess.py

Notes:
- This module uses NLTK for tokenization, stopwords and lemmatization.
- If NLTK resources are missing, the module will attempt to download them automatically (quietly).
- Functions are small and unit-test friendly.

Functions provided:
- normalize_text(text, keep_hyphen=False) -> str
- tokenize(text, remove_stopwords=True) -> List[str]
- tokenize_and_lemmatize(text, remove_stopwords=True) -> List[str]
- get_ngrams(tokens, n) -> List[str]
- simple_highlight(original_text, tokens_to_highlight) -> str  (returns a very small HTML-marked string)
"""

from typing import List, Iterable, Set
import re

# Try to import NLTK and ensure required resources exist; download quietly if missing.
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    # Ensure required data packages are present; download only if missing
    for pkg in ("wordnet", "omw-1.4", "punkt", "stopwords"):
        try:
            nltk.data.find(f"corpora/{pkg}" if pkg in ("wordnet", "stopwords", "omw-1.4") else f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

    _STOPWORDS = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()
    _HAS_NLTK = True
except Exception:
    # Fallbacks if NLTK can't be loaded (keeps module usable)
    _HAS_NLTK = False
    _STOPWORDS = set([
        "a","an","the","and","or","for","to","of","in","on","with","by","from","is","are","was","were","be","been","it","this",
        "that","these","those","as","at","but","if","into","about","between","through","during","before","after","above","below"
    ])
    # Minimal lemmatizer fallback: identity function
    class _DummyLemmatizer:
        def lemmatize(self, token: str) -> str:
            return token
    _LEMMATIZER = _DummyLemmatizer()

# regex to remove unwanted characters (keep alphanumeric and spaces)
_non_alphanum_re = re.compile(r"[^a-z0-9\s\-]")
_multiple_spaces_re = re.compile(r"\s+")
_digit_re = re.compile(r"\d+")


def normalize_text(text: str, keep_hyphen: bool = False, remove_digits: bool = False) -> str:
    """
    Normalize a string for phrase-level comparisons:
    - lowercases
    - optionally preserve hyphens (keep_hyphen=True)
    - removes non-alphanumeric characters (except spaces and optionally hyphens)
    - collapses multiple spaces to single space
    - optionally strips digits

    Args:
        text: raw input string
        keep_hyphen: if True, do not remove '-' characters (useful for tokens like 'rt-pcr')
        remove_digits: if True, remove digits

    Returns:
        normalized string
    """
    if not text:
        return ""

    text = text.lower()

    if remove_digits:
        text = _digit_re.sub(" ", text)

    if keep_hyphen:
        # temporarily replace hyphen with placeholder so global regex won't remove it
        text = text.replace("-", "<<HYPHEN>>")
        text = _non_alphanum_re.sub(" ", text)
        text = text.replace("<<HYPHEN>>", "-")
    else:
        text = _non_alphanum_re.sub(" ", text)

    text = _multiple_spaces_re.sub(" ", text).strip()
    return text


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Tokenize a string and optionally remove stopwords.
    Uses nltk.tokenize.word_tokenize if available; otherwise splits on whitespace.

    Args:
        text: input text (raw or normalized)
        remove_stopwords: whether to remove english stopwords

    Returns:
        list of token strings
    """
    if text is None:
        return []

    # prefer nltk tokenization if available
    if _HAS_NLTK:
        try:
            tokens = nltk.word_tokenize(text)
        except Exception:
            tokens = text.split()
    else:
        tokens = text.split()

    # normalize tokens (strip punctuation leftover) and filter empties
    tokens = [t.strip() for t in tokens if t.strip()]

    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS]

    return tokens


def tokenize_and_lemmatize(text: str, remove_stopwords: bool = True, keep_hyphen: bool = False) -> List[str]:
    """
    Full tokenization pipeline:
    - normalize_text (preserving hyphens if requested)
    - tokenization
    - lowercase already applied by normalize_text
    - remove stopwords (optional)
    - lemmatize each token

    Returns lemmatized token list (order preserved).
    """
    norm = normalize_text(text, keep_hyphen=keep_hyphen)
    tokens = tokenize(norm, remove_stopwords=remove_stopwords)
    lemmas = [_LEMMATIZER.lemmatize(t) for t in tokens]
    return lemmas


def get_ngrams(tokens: Iterable[str], n: int) -> List[str]:
    """
    Return n-grams (joined by space) for the provided token iterable.
    Example: tokens=['laser','ablation','kit'], n=2 -> ['laser ablation', 'ablation kit']

    Args:
        tokens: iterable of token strings
        n: size of ngram (2=bigram, 3=trigram)

    Returns:
        list of n-gram strings
    """
    toks = list(tokens)
    if n <= 0 or len(toks) < n:
        return []
    return [" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def simple_highlight(original_text: str, tokens_to_highlight: Iterable[str]) -> str:
    """
    A very small helper to return an HTML-ish version of original_text where each occurrence
    of tokens_to_highlight is wrapped with <mark> tags.
    - This is intentionally simple and case-insensitive.
    - Suitable for quick UI preview; NOT safe for raw HTML injection in production without escaping.

    Args:
        original_text: raw user-provided text
        tokens_to_highlight: iterable of token strings (normalized / lowercased preferred)

    Returns:
        string with <mark>token</mark> inserted around matches
    """
    if not original_text:
        return original_text

    out = original_text
    # do case-insensitive replace by using regex for each token
    for tok in sorted(set(tokens_to_highlight), key=len, reverse=True):
        if not tok:
            continue
        # escape regex meta in token
        safe_tok = re.escape(tok)
        # replace case-insensitive occurrences, wrap with <mark>
        out = re.sub(rf"(?i)\b{safe_tok}\b", lambda m: f"<mark>{m.group(0)}</mark>", out)
    return out


# Small demo usage when module run directly
if __name__ == "__main__":
    sample = "Laser for Varicose Veins & Proctology (RT-PCR, rapid test kit)"
    print("Original:", sample)
    print("Normalized:", normalize_text(sample, keep_hyphen=True))
    print("Tokens:", tokenize(sample))
    print("Lemmas:", tokenize_and_lemmatize(sample))
    print("Bigrams:", get_ngrams(tokenize_and_lemmatize(sample), 2))
    print("Highlighted:", simple_highlight(sample, ["laser", "proctology", "rt-pcr", "rapid"]))
