import pytest
import torch
import torch.nn as nn
from definition_64d1e54f783e45409443e33301dfd8fa import build_transformer_model

@pytest.fixture
def model_params():
    return {
        "vocab_size": 100,
        "d_model": 32,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 64,
    }

def test_build_transformer_model_output_type(model_params):
    model = build_transformer_model(**model_params)
    assert isinstance(model, nn.Module), "build_transformer_model should return a PyTorch nn.Module"

def test_build_transformer_model_vocab_size(model_params):
    model = build_transformer_model(**model_params)
    assert model.encoder.layers[0].self_attn.embed_dim == model_params["d_model"]

def test_build_transformer_model_num_layers(model_params):
    model = build_transformer_model(**model_params)
    assert len(model.encoder.layers) == model_params["num_layers"]

def test_build_transformer_model_num_heads(model_params):
    model = build_transformer_model(**model_params)
    assert model.encoder.layers[0].self_attn.num_heads == model_params["nhead"]

def test_build_transformer_model_forward_pass(model_params):
    model = build_transformer_model(**model_params)
    src = torch.randint(0, model_params["vocab_size"], (1, 10))  # Batch size 1, sequence length 10
    try:
        output = model(src, src)  # Pass source and target for simplicity
        assert output.shape == (1, 10, model_params["d_model"])
    except Exception as e:
        pytest.fail(f"Forward pass failed with exception: {e}")

