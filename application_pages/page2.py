import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from transformers import AutoTokenizer
import plotly.graph_objects as go

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

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.query = nn.Linear(embed_dim, head_dim)
        self.key = nn.Linear(embed_dim, head_dim)
        self.value = nn.Linear(embed_dim, head_dim)

    def forward(self, x, mask=None):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.heads = nn.ModuleList([SelfAttention(embed_dim, self.head_dim) for _ in range(num_heads)])
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        all_head_outputs = []
        all_head_weights = []
        for head in self.heads:
            output, weights = head(x, mask)
            all_head_outputs.append(output)
            all_head_weights.append(weights)

        concatenated_output = torch.cat(all_head_outputs, dim=-1)
        output = self.output_linear(concatenated_output)
        return output, all_head_weights

def run_page2():
    st.header("Multi-Head Attention Visualization")

    if 'input_sentence_mha' not in st.session_state:
        st.session_state['input_sentence_mha'] = ""
    if 'vocab_size_mha' not in st.session_state:
        st.session_state['vocab_size_mha'] = 50
    if 'max_length_mha' not in st.session_state:
        st.session_state['max_length_mha'] = 10
    if 'num_sentences_mha' not in st.session_state:
        st.session_state['num_sentences_mha'] = 100
    if 'num_heads_mha' not in st.session_state:
        st.session_state['num_heads_mha'] = 4
    if 'mha_attention_weights_display' not in st.session_state:
        st.session_state['mha_attention_weights_display'] = None
    if 'mha_token_labels_display' not in st.session_state:
        st.session_state['mha_token_labels_display'] = None

    with st.sidebar:
        st.header("Multi-Head Attention Parameters")
        input_sentence = st.text_area("Enter a sentence for MHA:", value=st.session_state['input_sentence_mha'], help="Enter the sentence to analyze. If empty, synthetic data will be used.")
        st.session_state['input_sentence_mha'] = input_sentence

        vocab_size = st.slider("Vocabulary Size (MHA)", min_value=10, max_value=100, value=st.session_state['vocab_size_mha'], help="Set the size of the vocabulary for synthetic data generation.")
        st.session_state['vocab_size_mha'] = vocab_size

        max_length = st.slider("Maximum Sentence Length (MHA)", min_value=5, max_value=20, value=st.session_state['max_length_mha'], help="Set the maximum length of generated sentences.")
        st.session_state['max_length_mha'] = max_length

        num_sentences = st.slider("Number of Sentences (MHA)", min_value=10, max_value=200, value=st.session_state['num_sentences_mha'], help="Set the number of sentences to generate.")
        st.session_state['num_sentences_mha'] = num_sentences

        num_heads = st.slider("Number of Attention Heads", min_value=1, max_value=8, value=st.session_state['num_heads_mha'], help="Set the number of attention heads for Multi-Head Attention.")
        st.session_state['num_heads_mha'] = num_heads

        run_analysis = st.button("Run MHA Analysis", help="Click to generate data and visualize multi-head attention.")

    if run_analysis:
        st.session_state['mha_attention_weights_display'] = None
        st.session_state['mha_token_labels_display'] = None

        with st.status("Generating data for MHA...", expanded=True) as status:
            token_labels = []
            if input_sentence:
                try:
                    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                    tokens = tokenizer.tokenize(input_sentence)
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                    synthetic_data = [token_ids]
                    token_labels = tokens
                    st.success("Sentence tokenized successfully.")
                except Exception as e:
                    st.error(f"Error tokenizing sentence: {e}")
                    status.update(label="Error during tokenization", state="error", expanded=False)
                    return
            else:
                synthetic_data = generate_synthetic_data(num_sentences, vocab_size, max_length)
                if synthetic_data:
                    token_labels = [str(token) for token in synthetic_data[0]]

            if not synthetic_data:
                st.warning("No data generated. Please adjust input parameters.")
                status.update(label="No data generated", state="warning", expanded=False)
                return

            actual_max_len = max(len(s) for s in synthetic_data) if synthetic_data else 0
            padding_length = max(max_length, actual_max_len) if input_sentence else max_length

            padded_data = []
            for sentence in synthetic_data:
                if len(sentence) < padding_length:
                    padded_data.append(sentence + [0] * (padding_length - len(sentence)))
                else:
                    padded_data.append(sentence[:padding_length])

            df = pd.DataFrame(padded_data, columns=[f"token_{i}" for i in range(padding_length)])

            if input_sentence and len(token_labels) < padding_length:
                token_labels = token_labels + ["<pad>"] * (padding_length - len(token_labels))
            elif not input_sentence:
                if synthetic_data and synthetic_data[0]:
                    token_labels = [str(token) for token in synthetic_data[0]]
                    if len(token_labels) < padding_length:
                         token_labels = token_labels + ["<pad>"] * (padding_length - len(token_labels))
                else:
                    token_labels = [str(i) for i in range(padding_length)]

            status.update(label="Data generation complete.", state="complete", expanded=False)

        st.subheader("Data Validation and Exploration")
        with st.expander("Show Data Validation Details (MHA)"):
            st.write("Missing values per column:")
            st.write(df.isnull().sum())
            st.write("Data types per column:")
            st.write(df.dtypes)
            if df.isnull().sum().sum() == 0:
                st.success("Validation successful: No missing values found in the dataset.")
            else:
                st.warning("Validation warning: Missing values detected in the dataset.")
            st.write("DataFrame head:")
            st.dataframe(df.head())
            st.write("DataFrame info:")
            from io import StringIO
            buffer = StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

        st.subheader("Summary Statistics (MHA)")
        with st.expander("Show Summary Statistics (MHA)"):
            st.dataframe(df.describe())

        st.subheader("Multi-Head Attention Visualization")
        with st.status("Calculating multi-head attention weights...", expanded=True) as status:
            embed_dim = 64 # Keep embed_dim consistent for now

            if embed_dim % num_heads != 0:
                st.error(f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads}). Please adjust.")
                status.update(label="Error: Invalid parameters", state="error", expanded=False)
                return

            embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

            mha_module = MultiHeadSelfAttention(embed_dim, num_heads)

            input_sequence_ids = torch.LongTensor(padded_data[0]).unsqueeze(0)
            embedded_sequence = embedding_layer(input_sequence_ids)

            mask = (input_sequence_ids != 0).unsqueeze(1)
            mask = mask.expand(-1, embedded_sequence.size(1), -1)

            try:
                _, all_attention_weights = mha_module(embedded_sequence, mask=mask)
                all_attention_weights_np = [weights.squeeze(0).detach().numpy() for weights in all_attention_weights]

                st.session_state['mha_attention_weights_display'] = all_attention_weights_np
                st.session_state['mha_token_labels_display'] = token_labels[:padding_length]
                status.update(label="Multi-Head Attention weights calculated.", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Error calculating multi-head attention weights: {e}")
                status.update(label="Error during attention calculation", state="error", expanded=False)
                st.session_state['mha_attention_weights_display'] = None
                st.session_state['mha_token_labels_display'] = None


        if st.session_state['mha_attention_weights_display'] is not None and st.session_state['mha_token_labels_display'] is not None:
            all_attention_weights_np = st.session_state['mha_attention_weights_display']
            token_labels_for_plot = st.session_state['mha_token_labels_display']

            tabs = st.tabs([f"Head {i+1}" for i in range(num_heads)])

            for i, weights_np in enumerate(all_attention_weights_np):
                with tabs[i]:
                    fig = go.Figure(data=go.Heatmap(
                        z=weights_np,
                        x=token_labels_for_plot,
                        y=token_labels_for_plot,
                        colorscale='Viridis',
                        colorbar=dict(title="Attention Weight")
                    ))

                    fig.update_layout(
                        title=f"Attention Head {i+1} Weight Heatmap",
                        xaxis_title="Key (Attended To)",
                        yaxis_title="Query (Attending From)",
                        xaxis_side="top",
                        yaxis=dict(autorange="reversed"),
                        height=600,
                        width=700
                    )
                    st.plotly_chart(fig)
        else:
            st.info("Run the MHA analysis to see the attention heatmaps.")

    st.markdown("""
    ## Multi-Head Attention: Diving Deeper into Transformer's Focus

    ### Business Value

    Multi-Head Attention is a crucial innovation in the Transformer architecture, significantly enhancing its ability to capture complex relationships within data. From a business perspective, this translates to:

    *   **Richer contextual understanding**: By attending to different parts of the input sequence simultaneously through multiple "heads," the model can learn diverse aspects of relationships. For instance, one head might focus on syntactic dependencies, while another captures semantic similarities.
    *   **Improved performance on diverse tasks**: This multi-faceted understanding leads to better performance in a wide range of NLP tasks like machine translation, text summarization, and question answering, as the model can leverage different types of information concurrently.
    *   **Robustness and flexibility**: Multi-head attention allows the model to be more robust to noisy data and more flexible in adapting to various linguistic nuances.

    ### Learning Goals

    -   Understand the concept of Multi-Head Attention as an extension of Self-Attention.
    -   Learn how multiple attention heads allow the model to capture different types of relationships simultaneously.
    -   Visualize the distinct attention patterns learned by individual heads.
    -   Appreciate how combining these heads leads to a more comprehensive understanding of the input sequence.

    ### Technical Explanation

    While Self-Attention allows a model to weigh the importance of different words, Multi-Head Attention takes this a step further by performing this attention mechanism multiple times in parallel. Each "head" is an independent self-attention module that learns to focus on different aspects of the input. The results from these separate attention heads are then concatenated and linearly transformed to produce the final output.

    The formula for Multi-Head Attention is given by:

    $$ \\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_h)W^O $$

    Where each $\\text{head}_i$ is computed as:

    $$ \\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$

    And:
    -   $h$: Represents the number of attention heads.
    -   $Q, K, V$: Are the Query, Key, and Value matrices, respectively, derived from the input embeddings.
    -   $W_i^Q, W_i^K, W_i^V$: Are learned weight matrices for each head $i$, which project the $Q, K, V$ into different, lower-dimensional subspaces. This allows each head to focus on different information.
    -   $\\text{Concat}(\\text{head}_1, ..., \\text{head}_h)$: Concatenates the outputs of all attention heads.
    -   $W^O$: Is a final learned linear projection that combines the concatenated outputs from all heads into the final Multi-Head Attention output.

    The key idea here is that by having multiple heads, the model can simultaneously attend to information from different representation subspaces at different positions. For example, one head might learn to attend to direct syntactic dependencies, while another might focus on more distant semantic relationships. This parallel processing of attention enriches the model's ability to understand the input comprehensively.

    In our visualization, you will see a separate heatmap for each attention head. Observe how each head might highlight different word-to-word relationships, demonstrating the diverse focus that Multi-Head Attention provides.
    """)

    st.markdown("""
    ### Understanding the Visualization

    Each heatmap represents the attention weights learned by a single attention head. The rows correspond to the "query" tokens (the token whose representation is being updated), and the columns correspond to the "key" tokens (the tokens that the query token is attending to). A brighter cell indicates a higher attention weight, meaning the query token is paying more attention to that particular key token.

    By comparing the heatmaps across different heads, you can observe:

    *   **Diverse Focus**: Different heads might focus on different linguistic phenomena. For example, one head might strongly attend to the next word in a sequence, while another might attend to a subject-verb pair regardless of distance.
    *   **Contextual Understanding**: The patterns reveal how the model is building contextual representations for each word by aggregating information from other words in the sentence.
    *   **Masking Effects (if applicable)**: In the context of a decoder or specific tasks, you might observe triangular patterns due to masking, where a word cannot attend to future words.

    This interactive visualization aims to demystify the inner workings of Multi-Head Attention, allowing you to gain intuitive insights into this powerful mechanism.
    """)