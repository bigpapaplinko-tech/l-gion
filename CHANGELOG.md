# Changelog

## 2026-10-17: Word database bootstrap + tokenizer improvements

This pass focused on making the project usable as a real seed for a resume job-fit database instead of only a prototype.

### Added
- `words.txt`: starter word list for common resume and technical vocabulary.
- `degrees.json`: initial keyword values from -3 to +3 to represent positive and negative resume signals.

### Improved
- `tokenizer.py` now keeps track of degree-only words separately from word-list words.
- Stats output is clearer about:
  - words loaded from the word list
  - degree-only terms
  - auto-registered words
  - overall degree count
- Interactive commands now include `:help`, `:words`, and `:lexicon` for easier debugging and database inspection.
- `:export PATH` allows writing the current vocabulary as JSON for downstream analysis.

### Why this matters
The repo is being designed around a larger idea: build a complete database of words and phrases, then compare resume language to job descriptions and estimate fit probabilities. These changes let the project start with a solid base instead of an empty lexicon.

### Next planned work
- expand `words.txt` into a much larger domain-specific dictionary
- separate words from phrases and multi-word patterns
- add weighting rules to tune the score model
- store resume/job-match data so probabilities can be calibrated from real outcomes
