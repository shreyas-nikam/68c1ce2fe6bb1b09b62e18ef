import pytest
import torch
from definition_95b11afe4abd49f49582a226a5165357 import calculate_attention_weights

@pytest.fixture
def sample_query_key_value():
    # Create sample query, key, and value matrices for testing
    batch_size = 2
    seq_len = 3
    d_model = 4
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    return query, key, value


def test_calculate_attention_weights_no_mask(sample_query_key_value):
    query, key, value = sample_query_key_value
    attention_weights = calculate_attention_weights(query, key, value, mask=None)
    assert attention_weights.shape == (query.size(0), query.size(1), key.size(1))


def test_calculate_attention_weights_with_mask(sample_query_key_value):
    query, key, value = sample_query_key_value
    mask = torch.ones(query.size(0), query.size(1), key.size(1)).bool()
    mask[:, :, 0] = False  # Mask the first position
    attention_weights = calculate_attention_weights(query, key, value, mask=mask)

    # Check that the masked positions have very low attention weights
    assert torch.all(attention_weights[:, :, 0] <= -1e9) #Masked values should be close to negative infinity


def test_calculate_attention_weights_different_lengths(sample_query_key_value):
    query, key, value = sample_query_key_value
    # Change key length
    key = torch.randn(query.size(0), query.size(1) + 1, query.size(2))
    value = torch.randn(query.size(0), query.size(1) + 1, query.size(2))
    attention_weights = calculate_attention_weights(query, key, value, mask=None)
    assert attention_weights.shape == (query.size(0), query.size(1), key.size(1))

def test_calculate_attention_weights_batched_mask(sample_query_key_value):
    query, key, value = sample_query_key_value
    batch_size = query.size(0)
    seq_len_q = query.size(1)
    seq_len_k = key.size(1)
    mask = torch.ones(batch_size, seq_len_q, seq_len_k).bool()
    mask[0, :, 0] = False # mask first position in first batch
    mask[1, :, 1] = False # mask second position in second batch

    attention_weights = calculate_attention_weights(query, key, value, mask)
    assert torch.all(attention_weights[0, :, 0] <= -1e9)
    assert torch.all(attention_weights[1, :, 1] <= -1e9)

def test_calculate_attention_weights_incorrect_input_types():
    with pytest.raises(TypeError):
        calculate_attention_weights("query", 123, [1,2,3], {1:"mask"})
