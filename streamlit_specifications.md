
## Streamlit Application Requirements Specification

### 1. Application Overview

This Streamlit application provides an interactive visualization of the self-attention mechanism within a Transformer model. Users can input a sentence and explore the attention weights between different words, understanding how the model captures relationships within the input sequence.

**Learning Goals:**

*   Understand how self-attention relates different positions of a single sequence to compute a representation of the sequence.
*   Learn how multi-head attention allows the model to attend to information from different representation subspaces at different positions.
*   Explore the effect of masking on the attention weights (if implemented as an extension).
*   See what relationships the model learned within the structure and context of the provided data.

### 2. User Interface Requirements

**Layout and Navigation Structure:**

*   The application should have a clear and intuitive layout.
*   A sidebar should contain input widgets and controls.
*   The main area should display the visualization and relevant explanations.

**Input Widgets and Controls:**

*   **Text Input:** A text area where the user can enter a sentence.
*   **Vocabulary Size Slider:** A slider to control the vocabulary size for synthetic data generation.  Default value is 50.
*   **Maximum Length Slider:** A slider to control the maximum sentence length.  Default value is 10.
*   **Number of Sentences Slider:** A slider to control the number of sentences generated. Default value is 100.
*   **Run Analysis Button:** A button to trigger the self-attention analysis and visualization based on the provided sentence or synthetic data parameters.

**Visualization Components:**

*   **Attention Weight Heatmap:** A heatmap visualizing the attention weights between different words in the input sentence.
*   **Data Table:** Display of the processed data including padding.
*   **Summary Statistics Table:** Display summary statistics of numeric data.

**Interactive Elements and Feedback Mechanisms:**

*   **Tooltips:** Provide tooltips for each input widget to explain its purpose.
*   **Clear titles, labeled axes, and legends** on visualizations.
*   **Status Messages:** Display status messages to inform the user about the progress of the analysis (e.g., "Generating synthetic data...", "Calculating attention weights...", "Displaying heatmap...").
*   If feasible provide highlighting functionality for specific tokens in the heatmap for ease of inspection.

### 3. Additional Requirements

**Annotation and Tooltip Specifications:**

*   Each input widget (text input, sliders) should have a tooltip that explains its purpose and how it affects the visualization.
*   The heatmap should have annotations to indicate the word each row and column represent.
*   Data tables should have column descriptions.

**State Management:**

*   Use `st.session_state` to persist the values of input widgets across reruns. This will prevent the application from resetting when the user interacts with it.

### 4. Notebook Content and Code Requirements

**Extracted Code Stubs:**

```python
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset

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
```

**Usage in Streamlit Application:**

1.  **Import Libraries:** Include the necessary import statements at the beginning of the Streamlit application.
2.  **Input Widgets:** Create Streamlit widgets for user input:

```python
# Initialize session state
if 'input_sentence' not in st.session_state:
    st.session_state['input_sentence'] = ""
if 'vocab_size' not in st.session_state:
    st.session_state['vocab_size'] = 50
if 'max_length' not in st.session_state:
    st.session_state['max_length'] = 10
if 'num_sentences' not in st.session_state:
    st.session_state['num_sentences'] = 100


with st.sidebar:
    st.header("Input Parameters")
    input_sentence = st.text_area("Enter a sentence:", value=st.session_state['input_sentence'], help="Enter the sentence to analyze.")
    st.session_state['input_sentence'] = input_sentence # update session state

    vocab_size = st.slider("Vocabulary Size", min_value=10, max_value=100, value=st.session_state['vocab_size'], help="Set the size of the vocabulary for synthetic data generation.")
    st.session_state['vocab_size'] = vocab_size

    max_length = st.slider("Maximum Sentence Length", min_value=5, max_value=20, value=st.session_state['max_length'], help="Set the maximum length of generated sentences.")
    st.session_state['max_length'] = max_length

    num_sentences = st.slider("Number of Sentences", min_value=10, max_value=200, value=st.session_state['num_sentences'], help="Set the number of sentences to generate.")
    st.session_state['num_sentences'] = num_sentences

    run_analysis = st.button("Run Analysis")
```

3.  **Data Generation/Loading:**

```python
if run_analysis:
    if input_sentence:
        # Tokenize the input sentence (using AutoTokenizer)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Or any other suitable tokenizer
        tokens = tokenizer.tokenize(input_sentence)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        synthetic_data = [token_ids] # Treat the tokenized sentence as synthetic data

    else:
        # Generate synthetic data
        synthetic_data = generate_synthetic_data(num_sentences, vocab_size, max_length)

    #Pad Sequences for DataFrame creation
    padded_data = [sentence + [0] * (max_length - len(sentence)) for sentence in synthetic_data]

    df = pd.DataFrame(padded_data, columns=[f"token_{i}" for i in range(max_length)])
```

4.  **Data Validation and Display:**

```python
    # Data Validation Checks and Display
    st.subheader("Data Validation and Exploration")

    # Check for missing values
    st.write("Missing values per column:")
    st.write(df.isnull().sum())

    # Check data types
    st.write("Data types per column:")
    st.write(df.dtypes)

    # Assert no missing values in critical fields (all fields in this case)
    if df.isnull().sum().sum() == 0:
        st.success("Validation successful: No missing values found in the dataset.")
    else:
        st.warning("Validation warning: Missing values detected in the dataset.")

    st.write("DataFrame head:")
    st.dataframe(df.head())

    st.write("DataFrame info:")
    st.text(df.info())
```

5.  **Self-Attention Implementation (Placeholder):**

```python
    # Placeholder for Self-Attention Calculation and Visualization
    st.subheader("Self-Attention Visualization")
    st.write("Self-attention implementation coming soon...")
    # TODO: Implement the self-attention mechanism and visualization here.
    # This will involve:
    # 1. Defining the self-attention layer.
    # 2. Calculating the attention weights.
    # 3. Visualizing the attention weights as a heatmap.
```

6.  **Markdown display:**

```python
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

    $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

    Where:
    -   $Q$ (Query): Represents the current word for which we are trying to compute an attention-weighted representation. It's a vector that queries other words for their relevance.
    -   $K$ (Key): Represents all other words in the input sequence. Each word has a key vector that indicates its content. The dot product of the query with a key determines their compatibility.
    -   $V$ (Value): Also represents all other words in the input sequence. Each word has a value vector that contains the actual information to be passed through. Once attention scores are calculated, they are used to weigh these value vectors.
    -   $d_k$: Is the dimension of the key vectors. Dividing by $\sqrt{d_k}$ is a scaling factor that helps to prevent the dot products from becoming too large, which could push the softmax function into regions with very small gradients.

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
```

**Important Notes:**

*   The self-attention implementation is a placeholder and needs to be implemented. This includes defining the self-attention layer, calculating attention weights, and creating the heatmap visualization.
*   The `AutoTokenizer` requires the `transformers` library to be installed. The selected tokenizer should be appropriate for the expected input sentences.
*   Error handling should be added to the code to handle potential exceptions (e.g., invalid input, missing libraries).
*   The code should be optimized for performance to meet the constraint of running in under 5 minutes on a mid-spec laptop.

