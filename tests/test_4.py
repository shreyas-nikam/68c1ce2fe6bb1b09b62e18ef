import pytest
import torch
from definition_8debfeeb003c4d71928f1e4007bc13b0 import mask_future_tokens

@pytest.mark.parametrize("batch, expected_shape", [
    (torch.tensor([[1, 2, 3, 0, 0]]), torch.Size([1, 5, 5])),
    (torch.tensor([[4, 5, 6], [7, 8, 9]]), torch.Size([2, 3, 3])),
    (torch.tensor([[1, 2, 3]]), torch.Size([1, 3, 3])),
    (torch.tensor([[1]]), torch.Size([1, 1, 1])),
    (torch.tensor([]), torch.Size([0, 0, 0])),
])
def test_mask_future_tokens_shape(batch, expected_shape):
    mask = mask_future_tokens(batch)
    assert mask.shape == expected_shape

@pytest.mark.parametrize("batch", [
    torch.tensor([[1, 2, 3, 0, 0]]),
    torch.tensor([[4, 5, 6], [7, 8, 9]]),
    torch.tensor([[1, 2, 3]]),
    torch.tensor([[1]]),
    torch.tensor([]),
])
def test_mask_future_tokens_values(batch):
    mask = mask_future_tokens(batch)
    seq_len = batch.size(1) if len(batch.shape) > 1 else 0 # Handling empty tensor case
    if seq_len > 0:
        for i in range(seq_len):
            for j in range(seq_len):
                if j > i:
                    assert mask[0, i, j] == float('-inf') if len(batch.shape) > 1 else mask[0,i,j] == float('-inf')
                else:
                    assert mask[0, i, j] == 0.0 if len(batch.shape) > 1 else mask[0,i,j] == 0.0
