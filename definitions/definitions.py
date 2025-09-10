import random

def generate_synthetic_data(num_sentences, vocab_size, max_length):
    """Generates synthetic sentences."""

    if not isinstance(num_sentences, int) or not isinstance(vocab_size, int) or not isinstance(max_length, int):
        raise TypeError("Inputs must be integers.")
    if num_sentences < 0 or vocab_size < 0 or max_length < 0:
        raise ValueError("Inputs must be non-negative.")

    data = []
    for _ in range(num_sentences):
        sentence_length = random.randint(1, max_length) if max_length > 0 else 0
        sentence = [random.randint(0, vocab_size - 1) for _ in range(sentence_length)]
        data.append(sentence)
    return data

import torch
import torch.nn as nn

def build_transformer_model(vocab_size, d_model, nhead, num_layers, dim_feedforward):
    """Defines the Transformer model architecture using PyTorch."""
    transformer = nn.Transformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dim_feedforward=dim_feedforward,
    )

    # Override the forward method to handle the vocabulary size
    def forward(src, tgt):
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        output = transformer(src, tgt)
        return torch.nn.functional.linear(output, torch.eye(vocab_size)).transpose(0, 1)

    transformer.forward = forward
    return transformer

def train_model(model, dataloader, optimizer, num_epochs):
    """Trains the Transformer model.

    Args:
        model: The Transformer model.
        dataloader: The data loader.
        optimizer: The optimizer.
        num_epochs: The number of epochs.
    """
    if not isinstance(dataloader, DataLoader):
        raise TypeError("dataloader must be a torch.utils.data.DataLoader object.")
    if optimizer is None:
        raise AttributeError("optimizer cannot be None.")

    criterion = nn.MSELoss()  # Define loss function

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_attention_weights(sentence, model, tokenizer, layer, head):
    """Extracts and visualizes attention weights.
    Args:
        sentence: Input sentence.
        model: Trained Transformer model.
        tokenizer: Tokenizer.
        layer: Layer index.
        head: Attention head index.
    """
    try:
        # Tokenize the sentence
        inputs = tokenizer(sentence, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Get the model's output with attention weights
        outputs = model(**inputs, output_attentions=True)
        
        # Extract attention weights from the specified layer and head
        attention = outputs.attentions[layer][0, head].detach().numpy()

        # Visualize the attention weights
        fig, ax = plt.subplots()
        im = ax.imshow(attention, cmap="viridis")

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_yticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens)
        ax.set_yticklabels(tokens)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        ax.set_title(f"Attention Weights (Layer {layer}, Head {head})")
        fig.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")

import torch

def mask_future_tokens(batch):
    """Implements masking to prevent the model from attending to future tokens during decoding."""
    sz = batch.size(1) if len(batch.shape) > 1 else 0
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    
    # Handle the case of an empty batch
    if len(batch.shape) == 0 or batch.shape[0] == 0:
        return torch.empty((0,0,0))

    mask = mask.unsqueeze(0).expand(batch.shape[0], sz, sz).to(batch.device)
    return mask

import torch

def calculate_attention_weights(query, key, value, mask):
    """Calculates attention weights using scaled dot-product attention."""
    d_k = key.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None:
        scores = scores.masked_fill(~mask, -1e9)
    attention_weights = torch.softmax(scores, dim=-1)
    return attention_weights

import torch
import torch.nn as nn
import torch.nn.functional as F

def multi_head_attention(query, key, value, num_heads, mask):
    """Performs multi-head attention.
    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        num_heads: Number of attention heads.
        mask: Mask tensor.
    Returns:
        Output tensor.
    """
    batch_size, seq_len, d_model = query.shape
    if num_heads <= 0:
        raise ValueError("Number of heads must be greater than 0.")

    if d_model % num_heads != 0:
        raise RuntimeError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

    head_dim = d_model // num_heads

    # Linear transformations
    q = nn.Linear(d_model, d_model)(query)
    k = nn.Linear(d_model, d_model)(key)
    v = nn.Linear(d_model, d_model)(value)

    # Split into heads
    q = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    # Scaled dot-product attention
    attn_output = scaled_dot_product_attention(q, k, v, mask)

    # Concatenate heads
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, d_model)

    # Output linear transformation
    output = nn.Linear(d_model, d_model)(attn_output)

    return output


def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculates scaled dot-product attention.
    Args:
        q: Query tensor.
        k: Key tensor.
        v: Value tensor.
        mask: Mask tensor.
    Returns:
        Output tensor.
    """
    d_k = q.size(-1)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
    attn_probs = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, v)
    return output

import torch

def create_masks(src, tgt, padding_value=0):
    """Creates source and target masks to prevent attending to padding tokens."""

    src_padding_mask = (src == padding_value).unsqueeze(1).unsqueeze(2) if (src == padding_value).any() else None
    tgt_padding_mask = (tgt == padding_value).unsqueeze(1).unsqueeze(2) if (tgt == padding_value).any() else None
    src_mask = None
    tgt_mask = None
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask