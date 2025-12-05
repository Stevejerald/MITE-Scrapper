# config.py — tuned defaults

# Weights
EXACT_PHRASE_WEIGHT = 3        # keep exact phrase strong
TOKEN_WEIGHT = 2               # increased: each token counts more
FUZZY_STRONG_WEIGHT = 2
FUZZY_WEAK_WEIGHT = 1

# Fuzzy thresholds
FUZZY_STRONG_THRESHOLD = 85
FUZZY_WEAK_THRESHOLD = 70

# Score normalization & limits
MAX_KEYWORDS_CONSIDERED = 30   # lower cap so denom isn't huge
RELEVANT_SCORE_THRESHOLD = 15  # lower threshold (more permissive)

# Misc
MAX_MATCHES_RETURN = 200

# in config.py
MIN_MEANINGFUL_TOKENS_FOR_RELEVANT = 2
SINGLE_TOKEN_MAX_SCORE = 10
GENERIC_TOKEN_FREQ_RATIO = 0.30
GENERIC_TOKEN_BLACKLIST = ["fiber","media","converter","device","system","module","unit"]
