import re
from collections import Counter

def tokenize(text: str):
    return re.findall(r"[a-zA-Z']+", text.lower())

_SPAM_WORDS = {
    "free", "winner", "win", "prize", "claim", "offer", "money", "urgent",
    "bonus", "click", "limited", "cash", "now"
}

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)

def extract_features(text: str, feature_names: list[str]) -> dict:
    tokens = tokenize(text)
    token_counts = Counter(tokens)

    # base features you already had
    feats = {}
    feats["char_count"] = len(text)
    feats["word_count"] = len(tokens)
    feats["exclam_count"] = text.count("!")
    feats["question_count"] = text.count("?")
    feats["digit_count"] = sum(ch.isdigit() for ch in text)
    feats["upper_count"] = sum(ch.isupper() for ch in text)

    # ---- ADD: dataset-aligned features ----
    # words = number of word tokens
    feats["words"] = len(tokens)

    # links = number of URLs
    feats["links"] = len(_URL_RE.findall(text))

    # capital_words = number of ALL-CAPS words (based on raw text tokens)
    raw_words = re.findall(r"[A-Za-z']+", text)
    feats["capital_words"] = sum(1 for w in raw_words if w.isupper() and len(w) > 1)

    # spam_word_count = count of spammy tokens (based on tokenized lowercase words)
    feats["spam_word_count"] = sum(token_counts[w] for w in _SPAM_WORDS)

    # keep your existing “word_*/w_* / token itself” feature support
    for f in feature_names:
        if f in feats:
            continue
        if f.startswith("word_"):
            word = f[len("word_"):].lower()
            feats[f] = token_counts[word]
        elif f.startswith("w_"):
            word = f[len("w_"):].lower()
            feats[f] = token_counts[word]
        else:
            feats[f] = token_counts.get(f.lower(), 0)

    return {f: feats.get(f, 0) for f in feature_names}

