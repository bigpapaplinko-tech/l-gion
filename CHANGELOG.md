# Changelog

## 2026-10-17: Word database bootstrap + tokenizer improvements

This pass focused on making the project usable as a real seed for a resume job-fit database instead of only a prototype.

### Added
- `words.txt`: starter word list for common resume and technical vocabulary.
- `degrees.json`: initial keyword values from -3 to +3 to represent positive and negative resume signals.
- `categories.json`: starter mapping of words to categories (technical, business, management, etc.).
- `weights.json`: starter per-word weight multipliers.

### Improved
- `tokenizer.py` now keeps track of degree-only words separately from word-list words.
- Stats output is clearer about:
  - words loaded from the word list
  - degree-only terms
  - auto-registered words
  - overall degree count
- Interactive commands now include `:help`, `:words`, and `:lexicon` for easier debugging and database inspection.
- `:export PATH` allows writing the current vocabulary as JSON for downstream analysis.
- Phrase (n-gram) extraction with configurable min/max n and phrase boost multiplier.
- Repeat-based weighting for tokens and phrase boost multiplier to favor multi-word expressions.

### Why this matters
The repo is being designed around a larger idea: build a complete database of words and phrases, then compare resume language to job descriptions and estimate fit probabilities. These changes let the project start with a solid base instead of an empty lexicon.

### Next planned work
- expand `words.txt` into a much larger domain-specific dictionary
- add persistent phrase-specific indexing and frequency counters (analytics)
- integrate a small local rule-based scoring engine so results are deterministic without an AI
- add tests and CI for tokenization, edge cases, and export
