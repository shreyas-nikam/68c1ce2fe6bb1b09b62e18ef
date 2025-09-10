import pytest
from definition_1cb60e4e5d5042d5b10c2a8f13f4cab7 import generate_synthetic_data

@pytest.mark.parametrize("num_sentences, vocab_size, max_length, expected_type", [
    (10, 100, 20, list),
    (0, 50, 10, list),
    (5, 200, 5, list),
])
def test_generate_synthetic_data_valid(num_sentences, vocab_size, max_length, expected_type):
    data = generate_synthetic_data(num_sentences, vocab_size, max_length)
    assert isinstance(data, expected_type)
    assert len(data) == num_sentences
    for sentence in data:
        assert isinstance(sentence, list)
        assert len(sentence) <= max_length
        for token in sentence:
            assert isinstance(token, int)
            assert 0 <= token < vocab_size

@pytest.mark.parametrize("num_sentences, vocab_size, max_length, expected_exception", [
    (-1, 100, 20, ValueError),
    (10, -1, 20, ValueError),
    (10, 100, -1, ValueError),
    (10, 100.5, 20, TypeError),
])
def test_generate_synthetic_data_invalid_input(num_sentences, vocab_size, max_length, expected_exception):
    with pytest.raises(expected_exception):
        generate_synthetic_data(num_sentences, vocab_size, max_length)
