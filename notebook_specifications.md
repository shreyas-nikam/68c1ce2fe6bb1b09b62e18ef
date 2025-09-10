
## Notebook Overview

This Jupyter Notebook visualizes the self-attention mechanism within a Transformer model, inspired by the paper "Attention is All You Need" by Vaswani et al. (2017). It allows users to input a sentence and explore the attention weights between different words, demonstrating how the model captures relationships within the input sequence.

### Learning Goals

- Understand how self-attention relates different positions of a single sequence to compute a representation of the sequence.
- Learn how multi-head attention allows the model to attend to information from different representation subspaces at different positions.
- Explore the effect of masking on the attention weights in the decoder stack.
- See what relationships the model learned within the structure and context of the provided data.

## Code Requirements

### Expected Libraries

-   **torch**: Deep learning framework for building and training the Transformer model.
-   **torch.nn**: Neural network modules for defining the model architecture.
-   **torch.optim**: Optimization algorithms for training the model.
-   **numpy**: Numerical computation library for handling arrays and matrices.
-   **matplotlib.pyplot**: Plotting library for visualizing attention weights.
-   **seaborn**: Statistical data visualization library for enhanced heatmaps.
-   **ipwidgets**: Interactive widgets for user input and model control.
-   **transformers**: HuggingFace's transformers library for easy access to pre-trained models and tokenizers.
-   **pandas**: Data manipulation and analysis

### Algorithms and Functions to be Implemented

-   **`generate_synthetic_data(num_sentences, vocab_size, max_length)`**: Generates a synthetic dataset of sentences for demonstration purposes.
-   **`build_transformer_model(vocab_size, d_model, nhead, num_layers, dim_feedforward)`**: Defines the Transformer model architecture using PyTorch.
-   **`train_model(model, dataloader, optimizer, num_epochs)`**: Trains the Transformer model on the provided dataset.
-   **`visualize_attention_weights(sentence, model, tokenizer, layer, head)`**: Extracts and visualizes the attention weights for a given sentence, layer, and attention head.
-   **`mask_future_tokens(batch)`**: Implements masking to prevent the model from attending to future tokens during decoding.
-   **`calculate_attention_weights(query, key, value, mask=None)`**: Calculates the attention weights using the scaled dot-product attention mechanism.
-   **`multi_head_attention(query, key, value, num_heads, mask=None)`**: Performs multi-head attention by splitting the query, key, and value into multiple heads.
-   **`create_masks(src, tgt)`**: Creates the source and target masks to prevent attending to padding tokens.

### Visualizations

-   **Attention Weight Heatmap**: Displays the attention weights between words in a sentence for a selected layer and attention head.
-   **Line Plot**: Training loss curve to track model convergence.
-   **Bar Chart**: Comparing attention weights across different words.
-   **Dataframe**: Table showing summary data of the attention matrix.

## Notebook Sections (in Detail)

1.  **Introduction**

    *   Markdown Cell: Introduce the notebook, the self-attention mechanism, and the Transformer architecture. Explain the learning goals and the purpose of the visualization. Reference the original "Attention is All You Need" paper.
    $$
    \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    $$
    Where $Q$ is the query, $K$ is the key, $V$ is the value, and $d_k$ is the dimension of the key. Explain each of these terms.

2.  **Import Libraries**

    *   Code Cell: Import all necessary libraries: `torch`, `torch.nn`, `torch.optim`, `numpy`, `matplotlib.pyplot`, `seaborn`, `ipwidgets`, `transformers`, and `pandas`.
    *   Markdown Cell: Explain the purpose of each imported library.

3.  **Synthetic Data Generation**

    *   Markdown Cell: Explain the need for a synthetic dataset for demonstration and how it will be used. Describe the structure of the synthetic data, including the use of numerical, categorical, and optional time-series fields. State the content of the data to be included such as the vocabulary, sentence length, etc.
    *   Code Cell: Implement the `generate_synthetic_data(num_sentences, vocab_size, max_length)` function to create a synthetic dataset of sentences. The dataset should consist of tokenized sentences represented as numerical sequences.
    *   Code Cell: Execute the `generate_synthetic_data()` function to generate the dataset.
    *   Markdown Cell: Explain the generated dataset, including the number of sentences, vocabulary size, and maximum sentence length. Show a sample of the generated data. Provide details on data validation steps taken during generation.

4.  **Data Exploration and Preprocessing**

    *   Markdown Cell: Discuss data handling and validation. Confirm expected column names, data types, and the uniqueness of a primary key (if applicable). Assert no missing values in critical fields and log summary statistics for numeric columns.
    *   Code Cell: Load the synthetic dataset into a pandas DataFrame.
    *   Code Cell: Perform data validation checks (e.g., check for missing values, data types, and primary key uniqueness).
    *   Code Cell: Display summary statistics for numeric columns in the dataset.

5.  **Transformer Model Definition**

    *   Markdown Cell: Explain the Transformer model architecture, including the encoder, decoder, multi-head attention, and feed-forward networks. Show the formulas for multi-head attention:

    $$
        \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
    $$

    where

    $$
        \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    $$

    *   Code Cell: Implement the `build_transformer_model(vocab_size, d_model, nhead, num_layers, dim_feedforward)` function to define the Transformer model architecture using PyTorch.
    *   Markdown Cell: Explain the parameters of the `build_transformer_model()` function, such as `vocab_size`, `d_model`, `nhead`, `num_layers`, and `dim_feedforward`.

6.  **Model Training**

    *   Markdown Cell: Explain the training process, including the loss function, optimizer, and training loop.
    *   Code Cell: Implement the `train_model(model, dataloader, optimizer, num_epochs)` function to train the Transformer model on the provided dataset.
    *   Code Cell: Instantiate the Transformer model and optimizer.
    *   Code Cell: Execute the `train_model()` function to train the model.
    *   Markdown Cell: Explain the training progress, including the loss curve and training time.

7.  **Attention Weight Visualization Function**

    *   Markdown Cell: Explain how the attention weights will be extracted and visualized. Describe the format of the attention weights and how they represent the relationships between words.
    *   Code Cell: Implement the `visualize_attention_weights(sentence, model, tokenizer, layer, head)` function to extract and visualize the attention weights for a given sentence, layer, and attention head. This function should:
        1.  Tokenize the input sentence.
        2.  Pass the tokenized sentence through the Transformer model.
        3.  Extract the attention weights from the specified layer and attention head.
        4.  Display the attention weights as a heatmap using `matplotlib` or `seaborn`.
    * Markdown Cell: Describe the masking implementation of the tokens, and describe the function `mask_future_tokens(batch)`. Also describe the function `create_masks(src, tgt)` and how it creates padding tokens.

8.  **User Input and Interaction**

    *   Markdown Cell: Introduce the interactive widgets that will allow users to input sentences, select layers, and choose attention heads for visualization.
    *   Code Cell: Create interactive widgets using `ipwidgets` for sentence input, layer selection, and head selection.
    *   Markdown Cell: Explain how to use the interactive widgets to explore the attention weights.

9.  **Attention Weight Visualization with User Input**

    *   Code Cell: Use the `visualize_attention_weights()` function to display the attention weights based on the user's input from the interactive widgets.
    *   Code Cell: Implement masking functionality. Use the `mask_future_tokens(batch)` function in training to prevent attending to future tokens.
    *   Markdown Cell: Explain the displayed attention weights and how they reflect the model's understanding of the relationships between words in the input sentence.

10. **Multi-Head Attention Visualization**

    *   Markdown Cell: Explain how multi-head attention allows the model to attend to different aspects of the input sequence.
    *   Code Cell: Display the attention weights for different attention heads in the same layer to demonstrate the diversity of attention patterns.
    *   Markdown Cell: Analyze the different attention patterns and discuss what aspects of the input sequence each head might be attending to.

11. **Layer Selection Visualization**

    *   Markdown Cell: Explain how the attention weights change across different layers of the Transformer model.
    *   Code Cell: Display the attention weights for the same sentence and attention head in different layers to show how the attention patterns evolve.
    *   Markdown Cell: Analyze the evolution of attention patterns across layers and discuss how the model refines its understanding of the input sequence.

12. **Quantitative Analysis of Attention**

    *   Markdown Cell: Introduce a method for quantitative analysis of attention weights, for example, calculating the average attention weight for each word.
    *   Code Cell: Calculate and display the average attention weights for each word in the input sentence. Use a bar chart to visualize these weights.
    *   Markdown Cell: Interpret the results of the quantitative analysis and discuss the relative importance of different words in the sentence.

13. **Attention Weight Statistics Table**

    *   Markdown Cell: Explain the benefits of viewing the attention matrix as a table.
    *   Code Cell: Create a pandas DataFrame from the attention matrix.
    *   Code Cell: Calculate descriptive statistics (mean, standard deviation, min, max) for each column (attention weight distribution for each word).
    *   Code Cell: Display the DataFrame with the calculated statistics.
    *   Markdown Cell: Analyze the statistics, providing insights into the distribution of attention weights for different words.

14. **Further Exploration and Exercises**

    *   Markdown Cell: Suggest further explorations and exercises for the user, such as:
        *   Experimenting with different input sentences.
        *   Comparing the attention patterns for different layers and attention heads.
        *   Analyzing the attention weights for specific linguistic phenomena (e.g., coreference resolution).
    *  Markdown Cell: Give the formula for calculating attention weights $Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$, and ask to implement it.

15. **Conclusion**

    *   Markdown Cell: Summarize the key learning points of the notebook and the insights gained from visualizing the self-attention mechanism.

16. **References**

    *   Markdown Cell: List the references used in the notebook, including the "Attention is All You Need" paper and any other relevant resources.
        *   Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in neural information processing systems*, *30*.
        *   Huggingface's Transformers library.
        *   Any other relevant libraries or resources.

