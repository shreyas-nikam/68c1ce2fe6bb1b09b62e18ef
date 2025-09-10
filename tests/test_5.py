import pytest
import torch
from definition_51739bc7d2114b6fabaa7bd04d9c7a04 import calculate_attention_weights

@pytest.fixture
def sample_query_key_value():
    # Create sample tensors for query, key, and value
    batch_size = 2
    seq_len = 3
    d_model = 4
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    return query, key, value

def test_calculate_attention_weights_no_mask(sample_query_key_value):
    query, key, value = sample_query_key_value
    attention_weights, attended_values = calculate_attention_weights(query, key, value, mask=None)
    assert attention_weights.shape == (query.shape[0], query.shape[1], key.shape[1])
    assert attended_values.shape == value.shape

def test_calculate_attention_weights_with_mask(sample_query_key_value):
    query, key, value = sample_query_key_value
    # Create a sample mask (e.g., to mask padding tokens)
    mask = torch.tensor([[False, False, True], [False, True, True]])
    attention_weights, attended_values = calculate_attention_weights(query, key, value, mask=mask)
    assert attention_weights.shape == (query.shape[0], query.shape[1], key.shape[1])
    assert attended_values.shape == value.shape

def test_calculate_attention_weights_query_key_value_different_lengths():
    batch_size = 2
    d_model = 4
    query = torch.randn(batch_size, 2, d_model)
    key = torch.randn(batch_size, 3, d_model)
    value = torch.randn(batch_size, 3, d_model)
    attention_weights, attended_values = calculate_attention_weights(query, key, value, mask=None)
    assert attention_weights.shape == (query.shape[0], query.shape[1], key.shape[1])
    assert attended_values.shape == value.shape

def test_calculate_attention_weights_zero_dimension():
     # Create sample tensors for query, key, and value
    batch_size = 0
    seq_len = 3
    d_model = 4
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    with pytest.raises(Exception):
        calculate_attention_weights(query, key, value, mask=None)

def test_calculate_attention_weights_invalid_mask_shape(sample_query_key_value):
    query, key, value = sample_query_key_value
    # Create an invalid mask with incorrect shape
    mask = torch.randn(2, 4)  # Incorrect shape
    with pytest.raises(RuntimeError): # Or a more specific exception based on your implementation
        calculate_attention_weights(query, key, value, mask=mask)
