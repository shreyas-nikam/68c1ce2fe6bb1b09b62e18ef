import pytest
from definition_06297da19b714ee2b26b83608f61b8e2 import build_transformer_model
import torch

@pytest.mark.parametrize("vocab_size, d_model, nhead, num_layers, dim_feedforward, expected_output", [
    (1000, 512, 8, 6, 2048, torch.nn.Transformer),
    (500, 256, 4, 3, 1024, torch.nn.Transformer),
    (2000, 128, 2, 2, 512, torch.nn.Transformer),
])
def test_build_transformer_model(vocab_size, d_model, nhead, num_layers, dim_feedforward, expected_output):
    model = build_transformer_model(vocab_size, d_model, nhead, num_layers, dim_feedforward)
    assert isinstance(model, expected_output)

@pytest.mark.parametrize("vocab_size, d_model, nhead, num_layers, dim_feedforward", [
    (100, 64, 4, 2, 256),
])
def test_transformer_model_output_shape(vocab_size, d_model, nhead, num_layers, dim_feedforward):
    model = build_transformer_model(vocab_size, d_model, nhead, num_layers, dim_feedforward)
    src = torch.randint(0, vocab_size, (10, 32))
    tgt = torch.randint(0, vocab_size, (10, 32))
    output = model(src, tgt)
    assert output.shape == (32, 10, vocab_size)

@pytest.mark.parametrize("vocab_size, d_model, nhead, num_layers, dim_feedforward", [
    (100, 64, 4, 2, 256),
])
def test_transformer_model_trainable_parameters(vocab_size, d_model, nhead, num_layers, dim_feedforward):
    model = build_transformer_model(vocab_size, d_model, nhead, num_layers, dim_feedforward)
    parameters = list(model.parameters())
    assert len(parameters) > 0


def test_build_transformer_model_invalid_input():
    with pytest.raises(TypeError):
        build_transformer_model("invalid", 512, 8, 6, 2048)

