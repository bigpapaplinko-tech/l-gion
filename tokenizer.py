#!/usr/bin/env python3
"""Token library: every character and every word gets a unique ID and a value.

This version adds:
- category loading (words -> categories)
- optional phrase (n-gram) extraction and registration
- per-word weight multipliers and repeat-based weighting
- export includes categories, weights, and phrase flags
- deterministic scoring engine (pre-AI): per-category buckets, raw and normalized score (0..100)
- phrase frequency counters and an :ingest command to update them from a directory of resumes

Run:
    python tokenizer.py --words words.txt --degrees degrees.json --categories categories.json --weights weights.json --vocab vocab.json --phrases --min-ngram 2 --max-ngram 4

Interactive commands: :help :stats :words :lexicon :save :export PATH :ingest PATH :score :quit
"""

import argparse
import json
import os
import re
import string
import sys
from dataclasses import dataclass, field
from collections import Counter, defaultdict

MIN_DEGREE, MAX_DEGREE = -3, 3
PAD = "<PAD>"
BASE_CHARS = (
    string.ascii_lowercase + string.ascii_uppercase + string.digits
    + string.punctuation + " \n\t"
)
# Letters/digits in any script, with inner apostrophes or hyphens kept ("don't", "full-stack").
WORD_RE = re.compile(r"[^\W_]+(?:['\u2019-][^\W_]+)*", re.UNICODE)


def default_char_value(ch):
    """Placeholder char values: a=1..z=26, digits = their number, everything else 0."""
    c = ch.lower()
    if "a" <= c <= "z":
        return ord(c) - 96
    if c in "0123456789":
        return int(c)
    return 0


@dataclass
class WordToken:
    text: str
    id: int
    char_ids: list
    value: float
    degree: int
    known: bool  # True if the word came from the loaded word list
    is_phrase: bool = False
    weight: float = 1.0


@dataclass
class Encoding:
    char_ids: list                      # every character of the input, in order
    words: list = field(default_factory=list)  # list[WordToken]
    total_value: float = 0.0            # weighted sum of token values
    total_degree: int = 0               # sum of token degrees (weighted by weight)
    new_words: int = 0                  # words registered during this call

    def to_ai_payload(self):
        """What would go to the AI step later."""
        return {
            "char_ids": self.char_ids,
            "words": [
                {
                    "word": w.text,
                    "id": w.id,
                    "value": w.value,
                    "degree": w.degree,
                    "is_phrase": w.is_phrase,
                    "weight": w.weight,
                }
                for w in self.words
            ],
            "total_value": self.total_value,
            "total_degree": self.total_degree,
        }


class Tokenizer:
    def __init__(self):
        self.char_to_id = {PAD: 0}
        self.word_to_id = {PAD: 0}
        self.seed_words = set()
        self.degree_words = set()
        self.char_values = {}   # overrides for default_char_value
        self.degrees = {}
        self.categories = {}    # word -> [categories]
        self.weights = {}       # word -> multiplier (float)
        # phrase settings (not persisted as runtime config)
        self.phrases_enabled = False
        self.min_ngram = 2
        self.max_ngram = 4
        self.repeat_bonus = 0.1
        self.phrase_boost = 1.5
        # phrase frequency counters persisted by user --phrase-freq
        self.phrase_freq = Counter()
        for ch in BASE_CHARS:
            self.char_id(ch)

    # ---- characters ----
    def char_id(self, ch):
        if ch not in self.char_to_id:
            self.char_to_id[ch] = len(self.char_to_id)
        return self.char_to_id[ch]

    def char_value(self, ch):
        if ch in self.char_values:
            return self.char_values[ch]
        if ch.lower() in self.char_values:
            return self.char_values[ch.lower()]
        return default_char_value(ch)

    # ---- words ----
    def word_id(self, word):
        key = word.casefold()
        if key not in self.word_to_id:
            self.word_to_id[key] = len(self.word_to_id)
        return self.word_to_id[key]

    def word_value(self, word):
        # default word value: sum of character values of the word (casefolded)
        return sum(self.char_value(c) for c in word.casefold())

    def load_words(self, path):
        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w:
                    self.word_id(w)
                    self.seed_words.add(w.casefold())
                    count += 1
        return count

    def load_degrees(self, path):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        for word, value in raw.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Degree for '{word}' must be a number, got {value!r}")
            key = word.casefold()
            self.degrees[key] = max(MIN_DEGREE, min(MAX_DEGREE, value))
            self.word_id(word)
            self.degree_words.add(key)

    def load_categories(self, path):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        for word, cats in raw.items():
            key = word.casefold()
            if isinstance(cats, str):
                cats = [cats]
            if not isinstance(cats, (list, tuple)):
                raise ValueError(f"Categories for '{word}' must be a string or list, got {cats!r}")
            self.categories[key] = [str(c) for c in cats]
            self.word_id(word)

    def load_weights(self, path):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        for word, value in raw.items():
            try:
                v = float(value)
            except Exception:
                raise ValueError(f"Weight for '{word}' must be numeric, got {value!r}")
            key = word.casefold()
            if v <= 0:
                raise ValueError(f"Weight for '{word}' must be > 0, got {v}")
            self.weights[key] = v
            self.word_id(word)

    def export_word_db(self, path):
        rows = []
        for word in sorted(self.word_to_id):
            if word == PAD:
                continue
            rows.append({
                "word": word,
                "id": self.word_to_id[word],
                "value": self.word_value(word),
                "degree": self.degrees.get(word, 0),
                "known": word in self.seed_words,
                "degree_only": word in self.degree_words and word not in self.seed_words,
                "categories": self.categories.get(word, []),
                "weight": self.weights.get(word, 1.0),
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        return len(rows)

    # ---- phrase frequency persistence ----
    def load_phrase_freq(self, path):
        if not os.path.exists(path):
            self.phrase_freq = Counter()
            return
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.phrase_freq = Counter({k: int(v) for k, v in data.items()})

    def save_phrase_freq(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(self.phrase_freq), f, ensure_ascii=False, indent=2)

    def update_phrase_freq_from_encoding(self, enc):
        for w in enc.words:
            if w.is_phrase:
                self.phrase_freq[w.text] += 1

    # ---- encoding ----
    def encode(self, text, *, phrases=None, min_ngram=None, max_ngram=None, repeat_bonus=None, phrase_boost=None):
        """Encode text into character ids and word/phrase tokens. Returns Encoding.

        Optional parameters override tokenizer defaults for this call.
        If phrases=True, n-gram phrases (min_ngram..max_ngram) are extracted and registered.
        """
        # runtime settings
        phrases_enabled = self.phrases_enabled if phrases is None else bool(phrases)
        min_n = self.min_ngram if min_ngram is None else int(min_ngram)
        max_n = self.max_ngram if max_ngram is None else int(max_ngram)
        repeat_b = self.repeat_bonus if repeat_bonus is None else float(repeat_bonus)
        p_boost = self.phrase_boost if phrase_boost is None else float(phrase_boost)

        enc = Encoding(char_ids=[self.char_id(ch) for ch in text])
        word_texts = []  # preserve sequence of word keys

        # extract words and create word tokens
        for m in WORD_RE.finditer(text):
            raw = m.group(0)
            key = raw.casefold()
            is_new = key not in self.word_to_id
            wid = self.word_id(raw)
            if is_new:
                enc.new_words += 1
            tok = WordToken(
                text=key,
                id=wid,
                char_ids=[self.char_id(c) for c in key],
                value=self.word_value(key),
                degree=self.degrees.get(key, 0),
                known=key in self.seed_words,
                is_phrase=False,
                weight=self.weights.get(key, 1.0),
            )
            enc.words.append(tok)
            word_texts.append(key)

        # optional phrase extraction (n-grams)
        if phrases_enabled and word_texts:
            n_words = len(word_texts)
            for n in range(min_n, min(max_n, n_words) + 1):
                for i in range(0, n_words - n + 1):
                    parts = word_texts[i : i + n]
                    phrase_text = " ".join(parts)
                    pkey = phrase_text.casefold()
                    is_new = pkey not in self.word_to_id
                    wid = self.word_id(phrase_text)
                    if is_new:
                        enc.new_words += 1
                    # phrase value: sum of component values * phrase_boost
                    comp_values = sum(self.word_value(w) for w in parts)
                    phrase_value = comp_values * p_boost
                    tok = WordToken(
                        text=pkey,
                        id=wid,
                        char_ids=[self.char_id(c) for c in pkey],
                        value=phrase_value,
                        degree=self.degrees.get(pkey, 0),
                        known=pkey in self.seed_words,
                        is_phrase=True,
                        weight=self.weights.get(pkey, 1.0) * p_boost,
                    )
                    enc.words.append(tok)

        # compute frequency-based multipliers and totals
        counts = Counter(w.text for w in enc.words)
        total_value = 0.0
        total_degree = 0
        for w in enc.words:
            freq = counts[w.text]
            repeat_multiplier = 1.0 + repeat_b * (freq - 1)
            multiplier = w.weight * repeat_multiplier
            total_value += w.value * multiplier
            # degrees are integer signals; apply weight multiplier when summing to allow intensity
            total_degree += int(round(w.degree * multiplier))
            # store effective weight for reporting/debugging
            w.weight = multiplier
        enc.total_value = total_value
        enc.total_degree = total_degree
        return enc

    # ---- deterministic scoring (pre-AI) ----
    def score_encoding(self, enc, *, max_value=1000.0, max_degree=50.0):
        """Compute deterministic score from an Encoding.

        Returns a dict with:
          - per_category: {category: value}
          - raw_value: float (sum of weighted token values)
          - raw_degree: int
          - category_breakdown: list of (category, value)
          - normalized_score: 0..100 (clipped)

        Normalization uses max_value and max_degree as expected maxima; these are heuristics
        you can tune as you gather data. If you want percentile calibration from real data,
        collect observed maxima and fit later.
        """
        # per-category accumulation
        per_cat = defaultdict(float)
        uncategorized = 0.0
        for w in enc.words:
            val = w.value * w.weight
            cats = self.categories.get(w.text, [])
            if cats:
                share = val / len(cats)
                for c in cats:
                    per_cat[c] += share
            else:
                uncategorized += val

        # raw totals
        raw_value = float(enc.total_value)
        raw_degree = int(enc.total_degree)

        # combine into a simple heuristic score: weighted sum of normalized value and degree
        v_score = max(0.0, min(1.0, raw_value / float(max_value)))
        d_score = max(-1.0, min(1.0, raw_degree / float(max_degree)))  # degree can be negative
        # map degree score (-1..1) to (0..1) where negative degrees reduce score
        d_score_mapped = (d_score + 1.0) / 2.0

        # combine with tunable weights
        VALUE_WEIGHT = 0.8
        DEGREE_WEIGHT = 0.2
        combined = VALUE_WEIGHT * v_score + DEGREE_WEIGHT * d_score_mapped
        normalized = max(0.0, min(100.0, combined * 100.0))

        return {
            "per_category": dict(per_cat),
            "uncategorized": uncategorized,
            "raw_value": raw_value,
            "raw_degree": raw_degree,
            "normalized_score": normalized,
            "category_breakdown": sorted(per_cat.items(), key=lambda kv: -kv[1]),
        }

    # ---- persistence ----
    def save(self, path):
        data = {
            "char_to_id": self.char_to_id,
            "word_to_id": self.word_to_id,
            "seed_words": sorted(self.seed_words),
            "degree_words": sorted(self.degree_words),
            "char_values": self.char_values,
            "degrees": self.degrees,
            "categories": self.categories,
            "weights": self.weights,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        t = cls()
        t.char_to_id = data["char_to_id"]
        t.word_to_id = data["word_to_id"]
        t.seed_words = set(data.get("seed_words", []))
        t.degree_words = set(data.get("degree_words", []))
        t.char_values = data.get("char_values", {})
        t.degrees = data.get("degrees", {})
        t.categories = data.get("categories", {})
        t.weights = {k: float(v) for k, v in data.get("weights", {}).items()}
        return t

    def stats(self):
        total_words = len(self.word_to_id) - 1
        degree_only = len(self.degree_words - self.seed_words)
        auto = total_words - len(self.seed_words) - degree_only
        phrase_count = sum(1 for w in self.word_to_id if " " in w)
        category_terms = len(self.categories)
        return {
            "characters": len(self.char_to_id) - 1,
            "words": total_words,
            "from_word_list": len(self.seed_words),
            "degree_only": degree_only,
            "auto_registered": auto,
            "degree_words": len(self.degrees),
            "phrase_terms": phrase_count,
            "category_terms": category_terms,
        }


def show(enc):
    print(f"\n  chars ({len(enc.char_ids)}): {enc.char_ids}")
    if not enc.words:
        print("  (no words)\n")
        return
    width = max(len(w.text) for w in enc.words)
    for w in enc.words:
        tag = "" if w.known else "  *auto"
        ptag = "(phrase)" if w.is_phrase else ""
        print(f"  {w.text:<{width}}  id={w.id:<7} value={w.value:<7.2f} degree={w.degree:+d} weight={w.weight:.2f} {ptag}{tag}")
    print(f"\n  total value: {enc.total_value:.2f}   total degree: {enc.total_degree:+d}")
    if enc.new_words:
        print(f"  {enc.new_words} new word(s) registered (*auto = not in the word list)")
    print()


def ingest_directory(tok: Tokenizer, path: str, *, phrases=True, min_ngram=None, max_ngram=None, phrase_freq_path=None):
    """Process all .txt files under path, update phrase frequency counters, and return number of files processed."""
    if not os.path.isdir(path):
        raise ValueError("ingest path must be a directory")
    min_n = min_ngram if min_ngram is not None else tok.min_ngram
    max_n = max_ngram if max_ngram is not None else tok.max_ngram
    files = 0
    for root, _, filenames in os.walk(path):
        for fn in filenames:
            if not fn.lower().endswith(".txt"):
                continue
            fp = os.path.join(root, fn)
            try:
                with open(fp, encoding="utf-8") as f:
                    text = f.read()
            except Exception:
                continue
            enc = tok.encode(text, phrases=phrases, min_ngram=min_n, max_ngram=max_n)
            tok.update_phrase_freq_from_encoding(enc)
            files += 1
    if phrase_freq_path:
        tok.save_phrase_freq(phrase_freq_path)
    return files


def main():
    p = argparse.ArgumentParser(description="Character + word token library with phrase, weighting, and deterministic scoring support")
    p.add_argument("--words", help="word list file, one word per line")
    p.add_argument("--degrees", help="JSON of word -> degree (-3..+3)")
    p.add_argument("--categories", help="JSON of word -> category or [categories]")
    p.add_argument("--weights", help="JSON of word -> multiplier (float)")
    p.add_argument("--vocab", help="vocab JSON to load if present and to write on :save")
    p.add_argument("--export", help="path to export the current word database as JSON")
    p.add_argument("--phrases", action="store_true", help="enable phrase (n-gram) extraction and registration")
    p.add_argument("--min-ngram", type=int, default=2, help="minimum n for phrase extraction (default 2)")
    p.add_argument("--max-ngram", type=int, default=4, help="maximum n for phrase extraction (default 4)")
    p.add_argument("--repeat-bonus", type=float, default=0.1, help="repeat word bonus multiplier per extra occurrence (default 0.1)")
    p.add_argument("--phrase-boost", type=float, default=1.5, help="phrase value boost multiplier (default 1.5)")
    p.add_argument("--phrase-freq", help="path to phrase frequency JSON file to load/save (default phrase_freq.json)")
    p.add_argument("--ingest", help="path to directory of .txt resumes to ingest phrase frequencies from")
    args = p.parse_args()

    try:
        if args.vocab and os.path.exists(args.vocab):
            tok = Tokenizer.load(args.vocab)
        else:
            tok = Tokenizer()
        if args.words:
            n = tok.load_words(args.words)
            print(f"Loaded {n} words from {args.words}")
        if args.degrees:
            tok.load_degrees(args.degrees)
            print(f"Loaded degree mappings from {args.degrees}")
        if args.categories:
            tok.load_categories(args.categories)
            print(f"Loaded categories from {args.categories}")
        if args.weights:
            tok.load_weights(args.weights)
            print(f"Loaded weights from {args.weights}")
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"Could not load: {e}", file=sys.stderr)
        sys.exit(1)

    # apply runtime phrase/weight settings
    tok.phrases_enabled = bool(args.phrases)
    tok.min_ngram = args.min_ngram
    tok.max_ngram = args.max_ngram
    tok.repeat_bonus = args.repeat_bonus
    tok.phrase_boost = args.phrase_boost

    phrase_freq_path = args.phrase_freq or "phrase_freq.json"
    # load phrase freq counters if present
    tok.load_phrase_freq(phrase_freq_path)

    if args.ingest:
        processed = ingest_directory(tok, args.ingest, phrases=tok.phrases_enabled, min_ngram=tok.min_ngram, max_ngram=tok.max_ngram, phrase_freq_path=phrase_freq_path)
        print(f"Ingested {processed} files from {args.ingest} and updated {phrase_freq_path}")

    if args.export:
        count = tok.export_word_db(args.export)
        print(f"Exported {count} words to {args.export}")

    print("Token library ready. :help :stats :words :lexicon :ingest PATH :save :export PATH :score :quit\n")
    last_encoding = None
    while True:
        try:
            text = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        cmd = text.strip()
        if cmd == ":quit":
            break
        if cmd == ":help":
            print("  :help         show available commands")
            print("  :stats        show vocabulary stats")
            print("  :words        list vocabulary entries")
            print("  :lexicon      list degree mappings")
            print("  :ingest PATH  ingest all .txt files under PATH and update phrase frequencies")
            print("  :save         write vocab file (requires --vocab)")
            print("  :export PATH  export the current word database to JSON")
            print("  :score        show deterministic score for last entered text")
            print("  :quit         exit\n")
            continue
        if cmd == ":stats":
            for k, v in tok.stats().items():
                print(f"  {k}: {v}")
            if tok.phrase_freq:
                top = tok.phrase_freq.most_common(10)
                print("\n  Top phrases (from phrase_freq):")
                for p, c in top:
                    print(f"    {p}: {c}")
            print()
            continue
        if cmd == ":words":
            for word in sorted(tok.word_to_id):
                if word == PAD:
                    continue
                known = "seed" if word in tok.seed_words else "auto"
                degree = tok.degrees.get(word, 0)
                cats = ",".join(tok.categories.get(word, []))
                weight = tok.weights.get(word, 1.0)
                print(f"  {word:<30} id={tok.word_to_id[word]:<6} {known:<6} degree={degree:+d} weight={weight:.2f} cats={cats}")
            print()
            continue
        if cmd in {":lexicon", ":degrees"}:
            if not tok.degrees:
                print("  (no degree mappings loaded)\n")
                continue
            for w, v in sorted(tok.degrees.items(), key=lambda kv: (-kv[1], kv[0])):
                print(f"  {w:<20} {v:+d}")
            print()
            continue
        if cmd.startswith(":ingest "):
            path = cmd[len(":ingest "):].strip()
            if not path:
                print("  Usage: :ingest PATH\n")
                continue
            try:
                processed = ingest_directory(tok, path, phrases=tok.phrases_enabled, min_ngram=tok.min_ngram, max_ngram=tok.max_ngram, phrase_freq_path=phrase_freq_path)
                print(f"  Ingested {processed} files from {path} and updated {phrase_freq_path}\n")
            except Exception as e:
                print(f"  Could not ingest: {e}\n")
            continue
        if cmd == ":save":
            if not args.vocab:
                print("  Start with --vocab path.json to enable saving.\n")
            else:
                tok.save(args.vocab)
                tok.save_phrase_freq(phrase_freq_path)
                print(f"  Saved to {args.vocab} and phrase freq to {phrase_freq_path}\n")
            continue
        if cmd.startswith(":export "):
            path = cmd[len(":export "):].strip()
            if not path:
                print("  Usage: :export PATH\n")
                continue
            count = tok.export_word_db(path)
            print(f"  Exported {count} words to {path}\n")
            continue
        if cmd == ":score":
            if last_encoding is None:
                print("  No text encoded yet. Type a line to encode and score it.\n")
                continue
            res = tok.score_encoding(last_encoding)
            print(f"  Deterministic normalized score: {res['normalized_score']:.1f}/100")
            print("  Top categories:")
            for c, v in res['category_breakdown'][:10]:
                print(f"    {c}: {v:.1f}")
            print()
            continue
        if not text:
            continue
        enc = tok.encode(text)
        last_encoding = enc
        show(enc)
    if args.vocab:
        tok.save(args.vocab)
        tok.save_phrase_freq(phrase_freq_path)


if __name__ == "__main__":
    main()
