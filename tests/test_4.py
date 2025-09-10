import pytest
import torch
from definition_a35035c9d200410c936b85302db6ade5 import mask_future_tokens

@pytest.mark.parametrize("batch, expected_shape", [
    (torch.randint(0, 10, (1, 5)), (1, 5, 5)),  # Single batch, sequence length 5
    (torch.randint(0, 10, (2, 3)), (2, 3, 3)),  # Batch size 2, sequence length 3
    (torch.randint(0, 10, (1, 1)), (1, 1, 1)),  # Single batch, sequence length 1 (edge case)
])
def test_mask_future_tokens_shape(batch, expected_shape):
    mask = mask_future_tokens(batch)
    assert mask.shape == expected_shape

@pytest.mark.parametrize("batch_size, seq_len", [(1, 5), (2, 3)])
def test_mask_future_tokens_values(batch_size, seq_len):
    batch = torch.randint(0, 10, (batch_size, seq_len))
    mask = mask_future_tokens(batch)

    for i in range(batch_size):
        for row in range(seq_len):
            for col in range(seq_len):
                if col > row:
                    assert mask[i, row, col] == float('-inf') , "Future tokens should be masked with -inf"
                else:
                    assert mask[i, row, col] == 0.0, "Past and current tokens should be 0"
