import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go

def get_positional_encoding(max_len, embed_dim):
    """
    Calculates positional encodings for a given maximum length and embedding dimension.
    Formula: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
             PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    if not isinstance(max_len, int) or not isinstance(embed_dim, int):
        raise TypeError("Inputs max_len and embed_dim must be integers.")
    if max_len <= 0 or embed_dim <= 0:
        raise ValueError("Inputs max_len and embed_dim must be positive.")

    pe = torch.zeros(max_len, embed_dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-(np.log(10000.0) / embed_dim)))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

def run_page3():
    st.header("Positional Encoding Visualization")

    if 'max_length_pe' not in st.session_state:
        st.session_state['max_length_pe'] = 20
    if 'embed_dim_pe' not in st.session_state:
        st.session_state['embed_dim_pe'] = 64
    if 'pe_display' not in st.session_state:
        st.session_state['pe_display'] = None

    with st.sidebar:
        st.header("Positional Encoding Parameters")
        max_length = st.slider("Maximum Sequence Length (PE)", min_value=10, max_value=100, value=st.session_state['max_length_pe'], help="Set the maximum sequence length for positional encoding.")
        st.session_state['max_length_pe'] = max_length

        embed_dim = st.slider("Embedding Dimension (PE)", min_value=16, max_value=256, value=st.session_state['embed_dim_pe'], step=16, help="Set the embedding dimension for positional encoding.")
        st.session_state['embed_dim_pe'] = embed_dim

        generate_pe_button = st.button("Generate Positional Encoding", help="Click to generate and visualize positional encodings.")

    if generate_pe_button:
        st.session_state['pe_display'] = None
        with st.status("Generating positional encodings...", expanded=True) as status:
            try:
                positional_encodings = get_positional_encoding(max_length, embed_dim)
                st.session_state['pe_display'] = positional_encodings.numpy()
                status.update(label="Positional encodings generated.", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Error generating positional encodings: {e}")
                status.update(label="Error during generation", state="error", expanded=False)

    if st.session_state['pe_display'] is not None:
        st.subheader("Positional Encoding Heatmap")

        fig = go.Figure(data=go.Heatmap(
            z=st.session_state['pe_display'],
            x=[f"Dim {i}" for i in range(st.session_state['pe_display'].shape[1])],
            y=[f"Pos {i}" for i in range(st.session_state['pe_display'].shape[0])],
            colorscale='RdBu',
            colorbar=dict(title="PE Value")
        ))

        fig.update_layout(
            title="Positional Encoding Values",
            xaxis_title="Embedding Dimension",
            yaxis_title="Position in Sequence",
            xaxis_side="bottom",
            height=600,
            width=800
        )
        st.plotly_chart(fig)

        st.markdown(f"""
        The heatmap above visualizes the positional encodings. Each row corresponds to a position in the sequence, and each column corresponds to a dimension in the embedding. Notice the distinct alternating sine and cosine patterns across the dimensions, which provide a unique encoding for each position.
        """)
    else:
        st.info("Adjust parameters in the sidebar and click 'Generate Positional Encoding' to see the visualization.")

    st.markdown("""
    ## Positional Encoding: Giving Transformers a Sense of Order

    ### Business Value

    Transformers, by their very nature, process sequences in parallel, which means they lose the inherent sequential order of words. Positional encoding addresses this crucial limitation, adding significant business value by:

    *   **Enabling sequence understanding**: For tasks like machine translation, text summarization, or speech recognition, the order of words is paramount. Positional encoding allows Transformers to understand "who did what to whom" and the temporal relationships between events.
    *   **Improving model accuracy**: By providing information about word positions, models can distinguish between sentences with the same words but different meanings due to word order (e.g., "dog bites man" vs. "man bites dog"). This leads to more accurate and reliable predictions.
    *   **Handling variable sequence lengths**: Positional encodings are designed to generalize to unseen sequence lengths, making the models flexible without needing retraining for different input sizes.

    ### Learning Goals

    -   Understand why positional encoding is necessary in Transformer models.
    -   Learn the mathematical formulas for generating sinusoidal positional encodings.
    -   Visualize how these encodings provide a unique position signal to each token.
    -   Grasp how positional encodings are combined with token embeddings to inject sequence order information.

    ### Technical Explanation

    Since the Transformer architecture does not inherently model sequence order (unlike recurrent neural networks), we need a way to inject information about the relative or absolute position of tokens in the sequence. Positional encodings serve this purpose. These are vectors that are added to the input embeddings at the bottom of the encoder and decoder stacks.

    The original Transformer paper uses sine and cosine functions of different frequencies to generate these encodings. The specific formulas for the positional encoding at position $pos$ and dimension $i$ are:

    $$ PE(pos, 2i) = \\sin\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right) $$

    $$ PE(pos, 2i+1) = \\cos\\left(\\frac{pos}{10000^{2i/d_{model}}}\\right) $$

    Where:
    -   $pos$: Represents the position of the token in the sequence (e.g., 0 for the first token, 1 for the second, and so on).
    -   $i$: Represents the dimension within the embedding vector. For a $d_{model}$-dimensional embedding, $i$ ranges from $0$ to $d_{model}/2 - 1$.
    -   $d_{model}$: Is the dimensionality of the model (i.e., the embedding dimension).

    This sinusoidal approach has a few key advantages:

    1.  **Unique Representation**: Each position gets a unique encoding.
    2.  **Generalization**: It can generalize to longer sequence lengths than those seen during training.
    3.  **Relative Positioning**: A linear transformation can represent a relative position, which is beneficial for the attention mechanism.

    By adding these positional encodings to the word embeddings, the Transformer can distinguish between words at different positions, allowing it to understand the grammatical structure and context that depends on word order.
    """)

    st.markdown("""
    ### Understanding the Positional Encoding Visualization

    The heatmap displays the values of the positional encoding matrix. Each row corresponds to a position in the sequence, and each column corresponds to an embedding dimension. You will observe:

    *   **Alternating Patterns**: The sine and cosine functions create distinct alternating patterns across the dimensions. Dimensions with smaller $2i/d_{model}$ values (lower frequencies) change slowly across positions, while those with larger $2i/d_{model}$ values (higher frequencies) change more rapidly.
    *   **Unique Positional Signature**: When you look at any given row (a specific position), the combination of sine and cosine values across all dimensions creates a unique "signature" for that position. This signature is what the Transformer learns to associate with a particular position.
    *   **Consistency**: The patterns are consistent across different dimensions, but with varying frequencies, ensuring that the model receives rich information about each token's location.

    This visualization helps demystify how a seemingly simple mathematical function can encode complex sequential information, enabling the Transformer to understand the order of words in a sentence.
    """)