# l-gion
Ummm stop lookin at ts its privete

This project is centered on building a resume-focused keyword database and scoring pipeline.

## Current state (summary)
We have a working tokenizer and bootstrapped vocabulary with:
- word list, degree lexicon, categories, and weights
- phrase (n-gram) extraction and phrase frequency counters (ingest support)
- deterministic, pre-AI scoring engine that returns a normalized 0-100 score plus per-category breakdown

Next recommended steps: collect resume/job corpora via `:ingest`, tune max_value/max_degree for normalization using observed data, and add CI/tests for the full pipeline.
