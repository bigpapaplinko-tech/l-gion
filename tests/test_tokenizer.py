import json
from tokenizer import Tokenizer


def test_encode_and_score():
    tok = Tokenizer()
    tok.word_id("led")
    tok.word_id("python")
    tok.degrees["led"] = 2
    tok.degrees["python"] = 1
    enc = tok.encode("Led a python project")
    assert enc.words, "expected tokens"
    res = tok.score_encoding(enc)
    assert "normalized_score" in res
    assert 0.0 <= res["normalized_score"] <= 100.0


def test_phrase_candidate_scoring_and_prune():
    tok = Tokenizer()
    tok.phrase_freq["leadership experience"] = 4
    tok.phrase_freq["random noise"] = 1
    results = tok.candidate_phrases(limit=10, min_freq=1)
    assert results and results[0][0] == "leadership experience"
    pruned = tok.prune_phrase_candidates(min_count=2, min_score=0.0)
    assert "random noise" not in pruned
