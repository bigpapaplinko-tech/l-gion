# Handoff: Application Statistician (resume probability app)

## Goal
A fun app where a person inputs a resume (plus other info, ideally a target job posting), the text is tokenized, tokens carry values, an AI makes sense of it all, and the app outputs a probability plus what to do next.

## Current status
Terminal prototypes only. No AI step yet (by design: "for now just code").

| File | What it does |
|------|--------------|
| `tokenizer.py` | Character and word token library. Unique IDs for every character and word, combined values, +/- degrees, JSON persistence. |
| `keyword_scorer.py` | Earlier prototype: type text, each word is looked up in a small lexicon of -3..+3 values, prints per-word values and a total. |

The two scripts are **not merged yet**. `tokenizer.py` is the more complete one and already includes the degree idea from the scorer.

## How to run
Requires Python 3.9+, no third-party packages.

```
python tokenizer.py
python tokenizer.py --words words.txt --degrees degrees.json --vocab vocab.json
python keyword_scorer.py
python keyword_scorer.py --lexicon my.json
```

Prompt commands: `:stats`, `:save`, `:quit` (tokenizer); `:lexicon`, `:quit` (scorer).

## How tokenizer.py works
1. **Characters:** all printable ASCII plus newline and tab get fixed IDs up front. Any other character (accents, emoji, other scripts) is registered the first time it appears, so nothing is dropped.
2. **Words:** loaded from a word list (one per line, `--words`). Words not in the list are registered on the fly and marked `*auto`, so every input is processable. Matching is case-insensitive (casefolded). Inner apostrophes and hyphens stay inside a word ("don't", "full-stack").
3. **Values:** a word's value is the sum of its character values. Placeholder char values: a=1 through z=26, digits = their number, everything else 0. Override via `char_values` in the vocab JSON or by editing `default_char_value()`.
4. **Degrees:** each word can carry a -3 to +3 degree (the +/- keyword signal) from `--degrees degrees.json`, e.g. `{"led": 2, "fired": -3}`. Unlisted words are 0.
5. **Output:** `encode(text)` returns an `Encoding` with all character IDs, a list of word tokens (text, id, char_ids, value, degree, known), and totals. `to_ai_payload()` packages that as a dict for the future AI step.
6. **Persistence:** `--vocab vocab.json` loads IDs if the file exists and saves on `:save` and on exit, so IDs stay stable between runs.

## Placeholders and gaps
- The sandbox had no dictionary, so there is no complete word list. Supply a `words.txt` for "every word." Until then, auto-registration covers unseen words.
- Character values are placeholders (alphabet position). The user may have their own scheme.
- The built-in lexicon in `keyword_scorer.py` (led +2, built +2, increased +3, reduced +2, launched +2, managed +1, python +1, assisted -1, helped -1, responsible -1, gap -2, fired -3) is made-up stand-in data.
- Known quirk: words that appear only in the degrees file are counted as auto-registered in `:stats` rather than as word-list words.

## Design conclusions from planning
- **The probability is the hard part.** Without outcome data (who got interviews or offers), any number is a vibe. Early versions should present a "match score" with explanations, then earn the word "probability."
- **Planned pipeline:** resume plus job posting in; Claude extracts structured JSON (skills, years, titles, education, achievements, required vs. preferred); embeddings for semantic matching and rules for hard requirements; weighted scoring in plain code for consistency; gap analysis and rewrite suggestions; later, calibration.
- **If large real data becomes available (millions of rows):** it needs outcome labels including rejections (not just hires, to avoid survivorship bias). Applicant data is sensitive personal data (GDPR/CCPA, US hiring-discrimination law), so provenance matters; scraping LinkedIn violates its terms. Models trained on past hiring decisions inherit those biases, so strip names, gender signals, and proxies, and audit across groups. A probability would be conditional on a company, role, and period. Millions of rows suits gradient-boosted trees or logistic regression on LLM-extracted features plus embeddings, then calibration. Do not train an LLM from scratch. Feature-importance methods (e.g. SHAP) can drive the "what to do next" advice.
- **Privacy basics:** keep names, gender, age, and photos out of scoring; delete resumes after processing unless the user opts in.

## Suggested next steps
1. Provide a real `words.txt` and the character value scheme.
2. Fill in `degrees.json` with the real keyword +/- values.
3. Merge the scorer's behavior into `tokenizer.py` (or import it) so one tool shows tokens, values, and degree totals.
4. Add the AI step: send `to_ai_payload()` plus the target job posting to Claude and have it return structured judgments against a rubric.
5. Compute the score in plain code, then add a gap/next-steps report.
6. Later: collect outcomes and calibrate.

## Open questions
- Who is this for: job seekers checking their own odds, or recruiters screening candidates? (Affects fairness design a lot.)
- Is there a specific data source, or is the large dataset hypothetical for now?
- Should totals stay a simple sum, or add weighting such as multipliers for repeated words or phrases?

## Paste this into a new chat to continue
> I'm continuing a project called the application statistician: a resume-analysis app that tokenizes input, assigns +/- values to keywords, and later sends them to an AI to output a probability and next steps. I have `tokenizer.py` (character and word token library with IDs, combined values, and -3..+3 degrees) and `keyword_scorer.py` (simple terminal scorer). I'm attaching both plus HANDOFF.md. Please read HANDOFF.md first, then help me with the next steps.
