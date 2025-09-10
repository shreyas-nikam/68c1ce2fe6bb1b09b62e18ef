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

def run_page1():
    st.header("Self-Attention Mechanism Visualization")

    if 'input_sentence' not in st.session_state:
        st.session_state['input_sentence'] = ""
    if 'vocab_size' not in st.session_state:
        st.session_state['vocab_size'] = 50
    if 'max_length' not in st.session_state:
        st.session_state['max_length'] = 10
    if 'num_sentences' not in st.session_state:
        st.session_state['num_sentences'] = 100
    if 'attention_weights_display' not in st.session_state:
        st.session_state['attention_weights_display'] = None
    if 'token_labels_display' not in st.session_state:
        st.session_state['token_labels_display'] = None

    with st.sidebar:
        st.header("Input Parameters")
        input_sentence = st.text_area("Enter a sentence:", value=st.session_state['input_sentence'], help="Enter the sentence to analyze. If empty, synthetic data will be used.")
        st.session_state['input_sentence'] = input_sentence

        vocab_size = st.slider("Vocabulary Size", min_value=10, max_value=100, value=st.session_state['vocab_size'], help="Set the size of the vocabulary for synthetic data generation.")
        st.session_state['vocab_size'] = vocab_size

        max_length = st.slider("Maximum Sentence Length", min_value=5, max_value=20, value=st.session_state['max_length'], help="Set the maximum length of generated sentences.")
        st.session_state['max_length'] = max_length

        num_sentences = st.slider("Number of Sentences", min_value=10, max_value=200, value=st.session_state['num_sentences'], help="Set the number of sentences to generate.")
        st.session_state['num_sentences'] = num_sentences

        run_analysis = st.button("Run Analysis", help="Click to generate data and visualize self-attention.")

    if run_analysis:
        st.session_state['attention_weights_display'] = None
        st.session_state['token_labels_display'] = None

        with st.status("Generating data...", expanded=True) as status:
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
        with st.expander("Show Data Validation Details"):
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

        st.subheader("Summary Statistics")
        with st.expander("Show Summary Statistics"):
            st.dataframe(df.describe())

        st.subheader("Self-Attention Visualization")
        with st.status("Calculating attention weights...", expanded=True) as status:
            embed_dim = 64
            head_dim = 64

            embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

            attention_module = SelfAttention(embed_dim, head_dim)

            input_sequence_ids = torch.LongTensor(padded_data[0]).unsqueeze(0)
            embedded_sequence = embedding_layer(input_sequence_ids)

            mask = (input_sequence_ids != 0).unsqueeze(1)
            mask = mask.expand(-1, embedded_sequence.size(1), -1)

            try:
                _, attention_weights = attention_module(embedded_sequence, mask=mask)
                attention_weights_np = attention_weights.squeeze(0).detach().numpy()
                st.session_state['attention_weights_display'] = attention_weights_np
                st.session_state['token_labels_display'] = token_labels[:padding_length]
                status.update(label="Attention weights calculated.", state="complete", expanded=False)
            except Exception as e:
                st.error(f"Error calculating attention weights: {e}")
                status.update(label="Error during attention calculation", state="error", expanded=False)
                st.session_state['attention_weights_display'] = None
                st.session_state['token_labels_display'] = None


        if st.session_state['attention_weights_display'] is not None and st.session_state['token_labels_display'] is not None:
            attention_weights_np = st.session_state['attention_weights_display']
            token_labels_for_plot = st.session_state['token_labels_display']

            fig = go.Figure(data=go.Heatmap(
                z=attention_weights_np,
                x=token_labels_for_plot,
                y=token_labels_for_plot,
                colorscale='Viridis',
                colorbar=dict(title="Attention Weight")
            ))

            fig.update_layout(
                title="Self-Attention Weight Heatmap",
                xaxis_title="Key (Attended To)",
                yaxis_title="Query (Attending From)",
                xaxis_side="top",
                yaxis=dict(autorange="reversed"),
                height=600,
                width=700
            )
            st.plotly_chart(fig)
        else:
            st.info("Run the analysis to see the attention heatmap.")

    st.markdown("""
    ## Transformer Self-Attention Mechanism Visualization

    ## Notebook Overview

    This application visualizes the self-attention mechanism within a Transformer model, inspired by the paper "Attention is All You Need" by Vaswani et al. (2017). It allows users to input a sentence and explore the attention weights between different words, demonstrating how the model captures relationships within the input sequence.

    ### Business Value

    Understanding the self-attention mechanism is crucial for anyone working with modern NLP models. This application provides a hands-on, visual approach to demystify how Transformers process sequences, making it easier to:
    *   **Debug and interpret model behavior**: By seeing which words the model focuses on, we can gain insights into its decision-making process.
    *   **Improve model performance**: A deeper understanding of attention patterns can guide architectural improvements or fine-tuning strategies.
    *   **Educate and onboard**: This interactive visualization serves as an excellent educational tool for grasping complex Transformer concepts.

    ### Learning Goals

    -   Understand how self-attention relates different positions of a single sequence to compute a representation of the sequence.
    -   Learn how multi-head attention allows the model to attend to information from different representation subspaces at different positions.
    -   Explore the effect of masking on the attention weights in the decoder stack.
    -   See what relationships the model learned within the structure and context of the provided data.

    At the core of the self-attention mechanism is the following formula:

    $$ \\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V $$

    Where:
    -   $Q$ (Query): Represents the current word for which we are trying to compute an attention-weighted representation. It's a vector that queries other words for their relevance.
    -   $K$ (Key): Represents all other words in the input sequence. Each word has a key vector that indicates its content. The dot product of the query with a key determines their compatibility.
    -   $V$ (Value): Also represents all other words in the input sequence. Each word has a value vector that contains the actual information to be passed through. Once attention scores are calculated, they are used to weigh these value vectors.
    -   $d_k$: Is the dimension of the key vectors. Dividing by $\\sqrt{d_k}$ is a scaling factor that helps to prevent the dot products from becoming too large, which could push the softmax function into regions with very small gradients.

    This formula essentially calculates how much focus (attention) each word should place on every other word in the sequence, allowing the model to weigh the importance of different words when processing a particular word.
    """)

    st.markdown("""### Synthetic Data Generation: Business Value and Technical Implementation

    **Business Value**: In the early stages of model development, especially when dealing with complex architectures like Transformers, having a reliable and controlled dataset is crucial. Synthetic data allows us to:

    *   **Rapidly prototype and test**: Without waiting for large, real-world datasets to be collected and preprocessed, we can immediately start building and testing our model's core functionalities.
    *   **Isolate and debug specific components**: By controlling the characteristics of the data, we can more easily pinpoint issues related to the model's architecture or learning algorithm, rather than being sidetracked by data complexities.
    *   **Demonstrate concepts clearly**: For educational purposes, synthetic data with understandable patterns makes it easier to illustrate how the self-attention mechanism works without the noise and intricacies of natural language.

    **Technical Implementation**: The `generate_synthetic_data` function creates a simplified representation of sentences. In a real-world scenario, sentences would be composed of words from a natural language vocabulary. For this demonstration, we are representing words as numerical tokens. The structure of the synthetic data is as follows:

    *   **Numerical Sequences**: Each "sentence" is a list of integers, where each integer represents a "word" from a predefined vocabulary.
    *   **Vocabulary Size**: We define a `vocab_size`, which is the total number of unique "words" our model can understand. The generated integers will be within the range `[0, vocab_size - 1]`.
    *   **Maximum Sentence Length**: We also define a `max_length` to control the variability of sentence lengths. Each generated sentence will have a random length between 1 and `max_length`.

    This simplified approach allows us to focus purely on the attention mechanism without needing extensive natural language processing (NLP) pipelines, tokenizers, and large embedding layers, which would add unnecessary complexity for this visualization.""")

    st.markdown("""### Explaining the Generated Dataset and Data Validation

    This section elaborates on the synthetic dataset created for demonstrating the Transformer's self-attention mechanism. The `generate_synthetic_data` function successfully produced a collection of numerical sequences, each representing a simplified "sentence."

    -   **Number of Sentences**: We generated `100` distinct sentences. This provides a sufficient volume of data to simulate training and observe attention patterns.
    -   **Vocabulary Size**: The vocabulary consists of `50` unique tokens, ranging from `0` to `49`. This small vocabulary keeps the examples manageable and easy to interpret.
    -   **Maximum Sentence Length**: Each sentence has a variable length, up to a maximum of `10` tokens. This variation is important as it reflects the diverse lengths of real-world sentences.

    **Sample of Generated Data:**

    As seen in the output of the previous code cell, here's a representative sample of the generated sentences:

    ```
    Sample of generated data:
    Sentence 1: [32, 23, 10, 39, 44, 18, 25, 34]
    Sentence 2: [17, 3, 20]
    Sentence 3: [42, 1, 46, 28, 30, 24]
    Sentence 4: [36, 15, 27, 49, 13, 14, 21, 10, 31]
    Sentence 5: [46, 40, 2, 45]
    ```

    **Data Validation Steps During Generation:**

    During the generation process, several implicit validation steps were taken:

    1.  **Type Checking**: The `generate_synthetic_data` function includes checks to ensure that `num_sentences`, `vocab_size`, and `max_length` are all integers. This prevents common input errors.
    2.  **Value Range Checks**: It also validates that these input parameters are non-negative, preventing the creation of invalid datasets (e.g., negative number of sentences).
    3.  **Token Range**: Each token within a generated sentence is guaranteed to be within the `[0, vocab_size - 1]` range, ensuring that all tokens are part of the defined vocabulary.
    4.  **Sentence Length**: Each sentence length is randomly determined but strictly adheres to the `max_length` parameter, and is at least 1 (unless `max_length` is 0, in which case it is 0). This ensures no empty sentences or sentences exceeding the intended maximum length are generated.

    These checks contribute to the robustness of the synthetic data generation process, ensuring that the data is well-formed and suitable for subsequent model training and attention visualization.""")

    st.markdown("""### Data Exploration and Preprocessing: Business Value and Technical Implementation

    **Business Value**: Before feeding any data into a machine learning model, it's paramount to understand its characteristics and ensure its quality. Data exploration and preprocessing are critical for:

    *   **Ensuring data quality**: Identifying and handling missing values, incorrect data types, or inconsistent entries prevents downstream model errors and biases.
    *   **Gaining insights**: Understanding the distribution and relationships within the data can inform model architecture choices, hyperparameter tuning, and interpretation of results.
    *   **Model robustness**: Well-preprocessed data leads to more stable and reliable models that generalize better to unseen data.
    *   **Troubleshooting**: When a model behaves unexpectedly, a thorough understanding of the data is the first step in diagnosing the problem.

    **Technical Implementation**: In this section, we will load our synthetic dataset into a pandas DataFrame for easier manipulation and perform essential data validation checks. Although our synthetic data is generated with some inherent validation, demonstrating these steps is crucial for real-world scenarios.

    Our data validation will focus on:

    1.  **Loading Data**: Converting the list of synthetic sentences into a structured pandas DataFrame. This allows us to leverage pandas' powerful data handling capabilities.
    2.  **Handling Variable Lengths**: Since our synthetic sentences have varying lengths, we will pad them to a uniform `max_length` before converting them into tensors for the model. This is a common practice in NLP to create batches of sequences.
    3.  **Confirming Expected Column Names and Data Types**: Although our synthetic data is simple, in a real dataset, this would involve verifying that columns have appropriate names and that their data types (e.g., integer, float, object) are as expected.
    4.  **Checking for Missing Values**: Asserting that there are no missing values in critical fields. For our synthetic data, this means ensuring every token position has a value after padding.
    5.  **Logging Summary Statistics**: For numeric columns (our token IDs), we will display summary statistics (mean, standard deviation, min, max, quartiles). This helps us understand the distribution of token IDs within our dataset.""")