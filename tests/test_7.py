import pytest
from definition_92c3e8d45cb34a0ca5adeea03532a014 import create_masks
import torch

def test_create_masks_no_padding():
    src = torch.tensor([[1, 2, 3]])
    tgt = torch.tensor([[4, 5, 6]])
    src_mask, tgt_mask = create_masks(src, tgt)
    assert src_mask is None
    assert tgt_mask is None

def test_create_masks_src_padding():
    src = torch.tensor([[1, 2, 0]])
    tgt = torch.tensor([[4, 5, 6]])
    src_mask, tgt_mask = create_masks(src, tgt)
    assert torch.equal(src_mask, (src == 0).unsqueeze(1).unsqueeze(2))
    assert tgt_mask is None

def test_create_masks_tgt_padding():
    src = torch.tensor([[1, 2, 3]])
    tgt = torch.tensor([[4, 5, 0]])
    src_mask, tgt_mask = create_masks(src, tgt)
    assert src_mask is None
    assert torch.equal(tgt_mask, (tgt == 0).unsqueeze(1).unsqueeze(2))

def test_create_masks_both_padding():
    src = torch.tensor([[1, 0, 3]])
    tgt = torch.tensor([[4, 5, 0]])
    src_mask, tgt_mask = create_masks(src, tgt)
    assert torch.equal(src_mask, (src == 0).unsqueeze(1).unsqueeze(2))
    assert torch.equal(tgt_mask, (tgt == 0).unsqueeze(1).unsqueeze(2))

def test_create_masks_empty_tensor():
    src = torch.tensor([[]])
    tgt = torch.tensor([[]])
    src_mask, tgt_mask = create_masks(src, tgt)
    assert src.shape[1] == 0
    assert tgt.shape[1] == 0
    
