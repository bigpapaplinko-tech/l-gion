#!/usr/bin/env python3
"""Keyword scorer: terminal prototype.

Type text, each word is looked up in a lexicon of values from -3 to +3
(sign = good/bad, size = degree). Unknown words count as 0.

Run:
    python keyword_scorer.py                  # uses the built-in placeholder lexicon
    python keyword_scorer.py --lexicon my.json   # your own values: {"word": 2, "other": -1}

Commands inside the prompt:
    :lexicon   show all loaded words and values
    :quit      exit
"""

import argparse
import json
import re
import sys

MIN_DEGREE, MAX_DEGREE = -3, 3

# Placeholder values. Replace with yours (edit here or pass --lexicon).
DEFAULT_LEXICON = {
    "led": 2,
    "built": 2,
    "increased": 3,
    "reduced": 2,
    "launched": 2,
    "managed": 1,
    "python": 1,
    "assisted": -1,
    "helped": -1,
    "responsible": -1,
    "gap": -2,
    "fired": -3,
}


def load_lexicon(path):
    if not path:
        return dict(DEFAULT_LEXICON)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    lexicon = {}
    for word, value in raw.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"Value for '{word}' must be a number, got {value!r}")
        lexicon[word.lower()] = max(MIN_DEGREE, min(MAX_DEGREE, value))
    return lexicon


def split_words(text):
    """Stand-in for tokenization: lowercase words only."""
    return re.findall(r"[a-z0-9+#.'-]+", text.lower())


def score(words, lexicon):
    """Return (total, hits, unknown). hits is a list of (word, value)."""
    hits, unknown = [], []
    for w in words:
        if w in lexicon:
            hits.append((w, lexicon[w]))
        else:
            unknown.append(w)
    total = sum(v for _, v in hits)
    return total, hits, unknown


def to_ai_payload(hits, unknown):
    """Placeholder for the future step: this is what would go to the AI."""
    return {"scored": [{"word": w, "value": v} for w, v in hits], "unscored": unknown}


def fmt(value):
    return f"{value:+g}"


def bar(value):
    n = int(abs(value))
    return ("+" * n) if value > 0 else ("-" * n)


def show_result(text, lexicon):
    words = split_words(text)
    total, hits, unknown = score(words, lexicon)
    if not words:
        print("  (no words found)")
        return
    print()
    if hits:
        width = max(len(w) for w, _ in hits)
        for w, v in hits:
            print(f"  {w:<{width}}  {fmt(v):>3}  {bar(v)}")
    else:
        print("  No keywords matched.")
    print(f"\n  Total: {fmt(total)}  ({len(hits)} matched, {len(unknown)} unscored)")
    if unknown:
        print(f"  Unscored: {', '.join(sorted(set(unknown)))}")
    print()
    # Future: send to_ai_payload(hits, unknown) to the AI step here.


def main():
    parser = argparse.ArgumentParser(description="Terminal keyword scorer")
    parser.add_argument("--lexicon", help="path to JSON file of word -> value")
    args = parser.parse_args()

    try:
        lexicon = load_lexicon(args.lexicon)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"Could not load lexicon: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Keyword scorer ({len(lexicon)} words loaded). :lexicon to list, :quit to exit.\n")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == ":quit":
            break
        if text == ":lexicon":
            for w, v in sorted(lexicon.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"  {w:<15} {fmt(v)}")
            print()
            continue
        show_result(text, lexicon)


if __name__ == "__main__":
    main()
