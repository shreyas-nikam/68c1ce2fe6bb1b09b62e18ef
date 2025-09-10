import pytest
import torch
from definition_0b3e39df13444e5589cd888fbe379a18 import create_masks

def test_create_masks_no_padding():
    src = torch.tensor([[1, 2, 3]])
    tgt = torch.tensor([[4, 5, 6]])
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_masks(src, tgt)
    assert src_padding_mask is None
    assert tgt_padding_mask is None


def test_create_masks_with_padding():
    src = torch.tensor([[1, 2, 0]])
    tgt = torch.tensor([[4, 0, 6]])
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_masks(src, tgt)
    assert torch.equal(src_padding_mask, (src == 0).unsqueeze(1).unsqueeze(2))
    assert torch.equal(tgt_padding_mask, (tgt == 0).unsqueeze(1).unsqueeze(2))


def test_create_masks_empty_tensors():
    src = torch.tensor([[]])
    tgt = torch.tensor([[]])
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_masks(src, tgt)
    assert src_padding_mask is None
    assert tgt_padding_mask is None

def test_create_masks_different_padding_values():
    src = torch.tensor([[1, 2, -1]])
    tgt = torch.tensor([[4, -1, 6]])
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_masks(src, tgt, padding_value=-1)
    assert torch.equal(src_padding_mask, (src == -1).unsqueeze(1).unsqueeze(2))
    assert torch.equal(tgt_padding_mask, (tgt == -1).unsqueeze(1).unsqueeze(2))
    
def test_create_masks_various_sized_batches():
    src = torch.tensor([[1, 2, 3], [4, 5, 0]])
    tgt = torch.tensor([[4, 5, 0], [7, 8, 9]])
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_masks(src, tgt)
    assert torch.equal(src_padding_mask, (src == 0).unsqueeze(1).unsqueeze(2))
    assert torch.equal(tgt_padding_mask, (tgt == 0).unsqueeze(1).unsqueeze(2))

