"""
datastore.py

Loads keyword CSV files and builds searchable indexes for the matching engine.

Save as:
product_matcher/backend/app/matching/datastore.py
"""

from typing import List, Dict, Set
import pandas as pd
import os

from .preprocess import normalize_text, tokenize_and_lemmatize


class KeywordEntry:
    """
    Holds a single keyword phrase and its preprocessed variants.
    """
    def __init__(self, phrase: str, category: str):
        self.phrase: str = phrase.strip()
        self.category: str = category

        # Normalized text for exact phrase detection
        self.norm: str = normalize_text(self.phrase, keep_hyphen=True)

        # Tokenized + lemmatized for token/fuzzy matching
        self.tokens: Set[str] = set(tokenize_and_lemmatize(self.phrase, keep_hyphen=True))

    def __repr__(self):
        return f"KeywordEntry(phrase='{self.phrase}', category='{self.category}')"


class KeywordStore:
    """
    Stores all keyword entries from CSV files and provides:
      - entries[]: list of KeywordEntry
      - phrase_map: normalized phrase → KeywordEntry
      - token_index: token → set(indices of phrases containing that token)
    """
    def __init__(self):
        self.entries: List[KeywordEntry] = []
        self.phrase_map: Dict[str, KeywordEntry] = {}
        self.token_index: Dict[str, Set[int]] = {}

    # ----------------------------
    # Load CSV and populate store
    # ----------------------------
    def load_csv(self, path: str, category: str):
        """
        Load a CSV of phrases into the store.

        CSV must contain a column named 'phrase'.

        Args:
            path: string path of CSV file
            category: label e.g., 'Diagnostic', 'Endo'
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")

        df = pd.read_csv(path)

        if "phrase" not in df.columns:
            raise ValueError(f"CSV {path} missing 'phrase' column")

        for _, row in df.iterrows():
            phrase = str(row["phrase"]).strip()
            if not phrase:
                continue

            entry = KeywordEntry(phrase, category)
            idx = len(self.entries)

            # Add to master list
            self.entries.append(entry)

            # Add to phrase map
            self.phrase_map[entry.norm] = entry

            # Add tokens to token index
            for t in entry.tokens:
                if t not in self.token_index:
                    self.token_index[t] = set()
                self.token_index[t].add(idx)

    # ----------------------------
    # Accessors
    # ----------------------------
    def all_entries(self) -> List[KeywordEntry]:
        return self.entries

    def get_by_phrase(self, raw_phrase: str):
        """Return entry based on raw phrase (auto-normalized)."""
        norm = normalize_text(raw_phrase, keep_hyphen=True)
        return self.phrase_map.get(norm)

    def candidates_by_token(self, token: str) -> List[KeywordEntry]:
        """
        Return candidate entries that contain the given token.
        Used for narrowing fuzzy or full matches.
        """
        out = []
        if token in self.token_index:
            for idx in self.token_index[token]:
                out.append(self.entries[idx])
        return out

    def size(self) -> int:
        return len(self.entries)
