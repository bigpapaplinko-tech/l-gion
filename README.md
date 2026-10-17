# l-gion
Ummm stop lookin at ts its privete

This project is centered on building a resume-focused keyword database and scoring pipeline.
The goal is to tokenize resumes and job descriptions, track word and phrase signals, and estimate how well a candidate matches a target role based on weighted vocabulary and job-fit data.

## Current state
- `tokenizer.py` is the main tool for character and word tokenization.
- `keyword_scorer.py` remains as a simpler prototype for keyword value scoring.
- We now include starter data files to make it easier to build the database without starting from scratch.

## Improvements added
- Added `words.txt` with a starter list of common resume and technical terms.
- Added `degrees.json` with sample positive and negative keyword signals.
- Updated `tokenizer.py` to:
  - track degree-only words separately from dictionary words
  - improve stats reporting for auto-registered and degree-only words
  - add interactive commands: `:help`, `:words`, `:lexicon`, and `:export PATH`
  - support exporting a JSON word database for downstream analysis
- Documented the current direction and the planned scoring pipeline in `CHANGELOG.md`.

## Run locally
```bash
python tokenizer.py
python tokenizer.py --words words.txt --degrees degrees.json --vocab vocab.json
python keyword_scorer.py
python keyword_scorer.py --lexicon my.json
```

## Notes
The current scoring model is intentionally simple and uses placeholder character values plus word degrees. This keeps the project flexible while we build a larger, more grounded word and phrase database from real resume data and job postings.
