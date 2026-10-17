import json
from l_gion_tokenizer import Tokenizer


def test_encode_and_score():
    tok = Tokenizer()
    # seed a couple of words
    tok.word_id('led')
    tok.word_id('python')
    tok.degrees['led'] = 2
    tok.degrees['python'] = 1
    enc = tok.encode('Led a python project')
    assert enc.words, 'expected tokens'
    res = tok.score_encoding(enc)
    assert 'normalized_score' in res
    assert 0.0 <= res['normalized_score'] <= 100.0
