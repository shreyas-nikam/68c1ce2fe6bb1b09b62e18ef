import pytest
from definition_1e71c2ec9f37490c9fc9b887a9a13fc0 import generate_synthetic_data

@pytest.mark.parametrize("num_sentences, vocab_size, max_length, expected_type", [
    (5, 10, 8, list),
    (0, 5, 5, list),
    (1, 20, 15, list),
])
def test_generate_synthetic_data_returns_list(num_sentences, vocab_size, max_length, expected_type):
    data = generate_synthetic_data(num_sentences, vocab_size, max_length)
    assert isinstance(data, expected_type)

@pytest.mark.parametrize("num_sentences, vocab_size, max_length", [
    (5, 10, 8),
    (0, 5, 5),
    (1, 20, 15),
])
def test_generate_synthetic_data_list_length(num_sentences, vocab_size, max_length):
    data = generate_synthetic_data(num_sentences, vocab_size, max_length)
    assert len(data) == num_sentences

@pytest.mark.parametrize("vocab_size, max_length", [
    (10, 8),
    (5, 5),
    (20, 15),
])
def test_generate_synthetic_data_sentence_length(vocab_size, max_length):
    data = generate_synthetic_data(1, vocab_size, max_length)
    assert len(data[0]) <= max_length

@pytest.mark.parametrize("vocab_size, max_length", [
    (10, 8),
    (5, 5),
    (20, 15),
])
def test_generate_synthetic_data_token_range(vocab_size, max_length):
    data = generate_synthetic_data(5, vocab_size, max_length)
    for sentence in data:
        for token in sentence:
            assert 0 <= token < vocab_size

def test_generate_synthetic_data_invalid_input():
    with pytest.raises(TypeError):
        generate_synthetic_data("a", 10, 8)
