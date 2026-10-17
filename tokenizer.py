#!/usr/bin/env python3
"""Token library: every character and every word gets a unique ID and a value.

Layers
  1. Character tokens: all printable ASCII + newline/tab get IDs up front.
     Any other character (accents, emoji, other scripts) is registered the
     first time it shows up, so nothing typed is ever dropped.
  2. Word tokens: words come from a word list you load (one per line). Any word
     not in the list is registered on the fly, so coverage is always complete.
  3. Values: a word's value is the sum of its character values.
     Separately, a word can carry a "degree" from -3 to +3 (the +/- keyword
     signal from the scorer). Unlisted words have degree 0.

Run:
    python tokenizer.py
    python tokenizer.py --words words.txt --degrees degrees.json --vocab vocab.json

Files
  words.txt        one word per line (supply a full dictionary here)
  degrees.json     {"led": 2, "fired": -3}
  vocab.json       saved IDs so they stay stable between runs (--vocab loads it
                   if it exists and :save writes it)
  char values      edit default_char_value() or put "char_values" in vocab.json

Prompt commands: :help :stats :words :lexicon :save :export PATH :quit
"""

import argparse
import json
import os
import re
import string
import sys
from dataclasses import dataclass, field

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
    value: int
    degree: int
    known: bool  # True if the word came from the loaded word list


@dataclass
class Encoding:
    char_ids: list                      # every character of the input, in order
    words: list = field(default_factory=list)
    total_value: int = 0                # sum of word values
    total_degree: int = 0               # sum of +/- degrees
    new_words: int = 0                  # words registered during this call

    def to_ai_payload(self):
        """What would go to the AI step later."""
        return {
            "char_ids": self.char_ids,
            "words": [
                {"word": w.text, "id": w.id, "value": w.value, "degree": w.degree}
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
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        return len(rows)

    # ---- encoding ----
    def encode(self, text):
        enc = Encoding(char_ids=[self.char_id(ch) for ch in text])
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
            )
            enc.words.append(tok)
            enc.total_value += tok.value
            enc.total_degree += tok.degree
        return enc

    # ---- persistence ----
    def save(self, path):
        data = {
            "char_to_id": self.char_to_id,
            "word_to_id": self.word_to_id,
            "seed_words": sorted(self.seed_words),
            "degree_words": sorted(self.degree_words),
            "char_values": self.char_values,
            "degrees": self.degrees,
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
        return t

    def stats(self):
        total_words = len(self.word_to_id) - 1
        degree_only = len(self.degree_words - self.seed_words)
        auto = total_words - len(self.seed_words) - degree_only
        return {
            "characters": len(self.char_to_id) - 1,
            "words": total_words,
            "from_word_list": len(self.seed_words),
            "degree_only": degree_only,
            "auto_registered": auto,
            "degree_words": len(self.degrees),
        }


def show(enc):
    print(f"\n  chars ({len(enc.char_ids)}): {enc.char_ids}")
    if not enc.words:
        print("  (no words)\n")
        return
    width = max(len(w.text) for w in enc.words)
    for w in enc.words:
        tag = "" if w.known else "  *auto"
        print(f"  {w.text:<{width}}  id={w.id:<7} value={w.value:<5} degree={w.degree:+d}{tag}")
    print(f"\n  total value: {enc.total_value}   total degree: {enc.total_degree:+d}")
    if enc.new_words:
        print(f"  {enc.new_words} new word(s) registered (*auto = not in the word list)")
    print()


def main():
    p = argparse.ArgumentParser(description="Character + word token library")
    p.add_argument("--words", help="word list file, one word per line")
    p.add_argument("--degrees", help="JSON of word -> degree (-3..+3)")
    p.add_argument("--vocab", help="vocab JSON to load if present and to write on :save")
    p.add_argument("--export", help="path to export the current word database as JSON")
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
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"Could not load: {e}", file=sys.stderr)
        sys.exit(1)

    if args.export:
        count = tok.export_word_db(args.export)
        print(f"Exported {count} words to {args.export}")

    print("Token library ready. :help :stats :words :lexicon :save :export PATH :quit\n")
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
            print("  :save         write vocab file (requires --vocab)")
            print("  :export PATH  export the current word database to JSON")
            print("  :quit         exit\n")
            continue
        if cmd == ":stats":
            for k, v in tok.stats().items():
                print(f"  {k}: {v}")
            print()
            continue
        if cmd == ":words":
            for word in sorted(tok.word_to_id):
                if word == PAD:
                    continue
                known = "seed" if word in tok.seed_words else "auto"
                degree = tok.degrees.get(word, 0)
                print(f"  {word:<20} id={tok.word_to_id[word]:<6} {known:<8} degree={degree:+d}")
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
        if cmd == ":save":
            if not args.vocab:
                print("  Start with --vocab path.json to enable saving.\n")
            else:
                tok.save(args.vocab)
                print(f"  Saved to {args.vocab}\n")
            continue
        if cmd.startswith(":export "):
            path = cmd[len(":export "):].strip()
            if not path:
                print("  Usage: :export PATH\n")
                continue
            count = tok.export_word_db(path)
            print(f"  Exported {count} words to {path}\n")
            continue
        if not text:
            continue
        show(tok.encode(text))
    if args.vocab:
        tok.save(args.vocab)


if __name__ == "__main__":
    main()
