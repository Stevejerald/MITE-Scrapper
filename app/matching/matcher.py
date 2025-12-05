"""
matcher.py

Matcher with strict multi-word phrase handling:
- Multi-word phrases (e.g., "hospital info") must appear whole in the input to get phrase/token credit.
- Tokens that only appear inside multi-word CSV entries (and are not standalone CSV phrases) are ignored for token scoring.
- Keeps prior safeguards: IDF-weighting, generic token blacklist, single-char token ignore, single-token cap, fuzzy matching, dynamic denominator.

Save as:
product_matcher/backend/app/matching/matcher.py
"""

from typing import Dict, Any, List, Set
import math
from rapidfuzz import fuzz
from .datastore import KeywordStore, KeywordEntry
from .preprocess import normalize_text, tokenize_and_lemmatize
from . import config

# token idf/clamp settings
_MIN_IDF_FACTOR = 0.6
_MAX_IDF_FACTOR = 3.5
_MAX_TOKEN_WEIGHT = 10.0

# Safeguard defaults (can be mirrored in config.py)
_MIN_MEANINGFUL_TOKENS_FOR_RELEVANT = getattr(config, "MIN_MEANINGFUL_TOKENS_FOR_RELEVANT", 2)
_SINGLE_TOKEN_MAX_SCORE = getattr(config, "SINGLE_TOKEN_MAX_SCORE", 10)  # percent
_GENERIC_TOKEN_FREQ_RATIO = getattr(config, "GENERIC_TOKEN_FREQ_RATIO", 0.30)
_GENERIC_TOKEN_BLACKLIST = set(getattr(config, "GENERIC_TOKEN_BLACKLIST", ["fiber", "media", "converter", "system", "device", "unit", "module", "kit", "tool"]))


def _is_input_token_valid(tok: str) -> bool:
    """
    Input token validity filter:
    - Exclude tokens shorter than 2 characters (single letters)
    - Exclude pure numeric tokens
    - Accept tokens length >=2 and not purely numeric
    """
    if not tok:
        return False
    if len(tok) < 2:
        return False
    if tok.isdigit():
        return False
    return True


class Matcher:
    def __init__(self, store: KeywordStore):
        self.store = store

        # Build token role sets:
        # - single_word_tokens: tokens that exist as standalone phrases in the CSV
        # - multiword_tokens: tokens that appear in at least one multiword phrase
        single_tokens: Set[str] = set()
        multi_tokens: Set[str] = set()

        for entry in self.store.all_entries():
            toks = entry.tokens
            if not toks:
                continue
            if len(toks) == 1:
                # single-word phrase (e.g., "laser")
                single_tokens.update(toks)
            else:
                # multi-word phrase (e.g., "hospital info")
                multi_tokens.update(toks)

        self.single_word_tokens: Set[str] = single_tokens
        self.multiword_tokens: Set[str] = multi_tokens

    def analyze(self, text: str, category_filter: str = "all") -> Dict[str, Any]:
        if not text or not text.strip():
            return {"relevant": False, "score_pct": 0, "matches": [], "category_scores": {}, "raw_score": 0.0, "matched_count": 0}

        norm_text = normalize_text(text, keep_hyphen=True)

        # Tokenize + lemmatize full; then filter out single-char and numeric tokens for matching
        text_tokens_raw = list(set(tokenize_and_lemmatize(text, keep_hyphen=True)))
        text_tokens = set([t for t in text_tokens_raw if _is_input_token_valid(t)])

        matches: List[Dict[str, Any]] = []
        matched_phrases: Set[str] = set()
        raw_score = 0.0
        category_scores: Dict[str, float] = {}

        def _add_match(entry: KeywordEntry, match_type: str, weight: float, matched_text: str):
            nonlocal raw_score
            if entry.phrase in matched_phrases:
                return
            matches.append({
                "phrase": entry.phrase,
                "category": entry.category,
                "match_type": match_type,
                "weight": weight,
                "matched_text": matched_text
            })
            matched_phrases.add(entry.phrase)
            raw_score += weight
            category_scores[entry.category] = category_scores.get(entry.category, 0.0) + weight

        cf = (category_filter or "all").strip().lower()
        total_entries = max(1, self.store.size())

        # 1) Exact phrase matches (strong signal)
        for norm_phrase, entry in list(self.store.phrase_map.items()):
            if cf != "all" and entry.category.lower() != cf:
                continue
            if not norm_phrase:
                continue
            if norm_phrase in norm_text:
                _add_match(entry, "exact", config.EXACT_PHRASE_WEIGHT, entry.phrase)

        # 2) Token-level matches (IDF-weighted), with multiword-aware token filtering
        candidate_indices: Set[int] = set()
        for tok in text_tokens:
            if tok in self.store.token_index:
                candidate_indices.update(self.store.token_index[tok])

        if not candidate_indices:
            candidate_entries = [e for e in self.store.all_entries() if (cf == "all" or e.category.lower() == cf)]
        else:
            candidate_entries = [self.store.entries[i] for i in candidate_indices if (cf == "all" or self.store.entries[i].category.lower() == cf)]

        # Helper: detect generic tokens by frequency or blacklist
        def is_generic_token(tok: str) -> bool:
            if tok in _GENERIC_TOKEN_BLACKLIST:
                return True
            freq = len(self.store.token_index.get(tok, []))
            if freq / total_entries > _GENERIC_TOKEN_FREQ_RATIO:
                return True
            return False

        meaningful_tokens_matched: Set[str] = set()

        for entry in candidate_entries:
            if entry.phrase in matched_phrases:
                continue
            # intersection with filtered tokens only
            overlap = entry.tokens.intersection(text_tokens)
            if overlap:
                # Apply multiword-aware token filtering:
                # Keep token t only if either:
                #  - t is present as a single-word phrase in CSV (self.single_word_tokens), OR
                #  - t is NOT exclusively part of multiword-only phrases (i.e., not in self.multiword_tokens)
                allowed_overlap = set()
                for t in overlap:
                    if (t in self.single_word_tokens) or (t not in self.multiword_tokens):
                        allowed_overlap.add(t)

                if not allowed_overlap:
                    # No allowed tokens (they were only parts of multiword phrases and those full phrases didn't match)
                    continue

                token_weight_sum = 0.0
                for t in allowed_overlap:
                    freq = len(self.store.token_index.get(t, []))
                    idf_factor = 1.0 + math.log((total_entries) / (1 + freq)) if freq >= 0 else 1.0
                    idf_factor = max(_MIN_IDF_FACTOR, min(_MAX_IDF_FACTOR, idf_factor))
                    token_weight_sum += config.TOKEN_WEIGHT * idf_factor

                weight = min(token_weight_sum, _MAX_TOKEN_WEIGHT)
                _add_match(entry, "token", weight, ", ".join(sorted(allowed_overlap)))

                for t in allowed_overlap:
                    if not is_generic_token(t):
                        meaningful_tokens_matched.add(t)

        # 3) Fuzzy matching for remaining candidate entries
        checked_entries: Set[str] = set()
        for entry in candidate_entries:
            if entry.phrase in matched_phrases:
                continue
            if entry.phrase in checked_entries:
                continue
            checked_entries.add(entry.phrase)
            try:
                ratio = fuzz.token_set_ratio(entry.norm, norm_text)
            except Exception:
                ratio = fuzz.ratio(entry.norm, norm_text)
            if ratio >= config.FUZZY_STRONG_THRESHOLD:
                _add_match(entry, "fuzzy_strong", config.FUZZY_STRONG_WEIGHT, f"ratio:{ratio}")
            elif ratio >= config.FUZZY_WEAK_THRESHOLD:
                _add_match(entry, "fuzzy_weak", config.FUZZY_WEAK_WEIGHT, f"ratio:{ratio}")

        # 4) Light fuzzy across all entries if no candidates and small dataset
        if not candidate_entries:
            if total_entries <= config.MAX_KEYWORDS_CONSIDERED:
                for entry in self.store.all_entries():
                    if entry.phrase in matched_phrases:
                        continue
                    if cf != "all" and entry.category.lower() != cf:
                        continue
                    ratio = fuzz.token_set_ratio(entry.norm, norm_text)
                    if ratio >= config.FUZZY_STRONG_THRESHOLD:
                        _add_match(entry, "fuzzy_strong", config.FUZZY_STRONG_WEIGHT, f"ratio:{ratio}")
                    elif ratio >= config.FUZZY_WEAK_THRESHOLD:
                        _add_match(entry, "fuzzy_weak", config.FUZZY_WEAK_WEIGHT, f"ratio:{ratio}")

        # Optional category boost
        for cat in list(category_scores.keys()):
            count_matches_for_cat = sum(1 for m in matches if m["category"] == cat)
            if count_matches_for_cat >= 2:
                bonus = 0.5 * (count_matches_for_cat - 1)
                category_scores[cat] = category_scores.get(cat, 0.0) + bonus

        # 5) Score normalization with dynamic denominator
        matched_count = max(1, len(matched_phrases))
        base_for_denominator = max(3, matched_count)
        denom_keywords = min(config.MAX_KEYWORDS_CONSIDERED, base_for_denominator)
        max_possible = config.EXACT_PHRASE_WEIGHT * denom_keywords
        score_pct = int(min(100, round(100.0 * raw_score / max(1.0, max_possible))))

        # SAFEGUARD: single-token or only-generic tokens handling
        has_exact_or_strong = any(m["match_type"] in ("exact", "fuzzy_strong") for m in matches)
        meaningful_count = len(meaningful_tokens_matched)

        if meaningful_count < _MIN_MEANINGFUL_TOKENS_FOR_RELEVANT and not has_exact_or_strong:
            final_relevant = False
            final_score = min(score_pct, _SINGLE_TOKEN_MAX_SCORE)
        else:
            final_relevant = score_pct >= config.RELEVANT_SCORE_THRESHOLD
            final_score = score_pct

        matches_sorted = sorted(matches, key=lambda m: (-m["weight"], m["match_type"], m["phrase"]))
        category_scores_out = {k: (int(v) if float(v).is_integer() else round(v, 2)) for k, v in category_scores.items()}

        return {
            "relevant": bool(final_relevant),
            "score_pct": int(final_score),
            "matches": matches_sorted,
            "category_scores": category_scores_out,
            "raw_score": raw_score,
            "matched_count": len(matched_phrases),
            "meaningful_tokens_matched": sorted(list(meaningful_tokens_matched)),
            "text_tokens_raw": sorted(text_tokens_raw),
            "text_tokens_filtered": sorted(list(text_tokens))
        }
