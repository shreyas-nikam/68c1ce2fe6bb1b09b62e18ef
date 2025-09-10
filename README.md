# QuLab: Transformer Attention & Positional Encoding Visualizer

![QuLab Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Description

QuLab is an interactive Streamlit application designed as a lab project to demystify the core mechanisms of Transformer models. Inspired by the seminal paper "Attention Is All You Need," this application provides a hands-on platform to explore the self-attention mechanism, multi-head attention, and positional encodings. Users can generate synthetic data or input custom sentences, visualize attention weights through heatmaps, and understand how Transformers process sequences by weighing the importance of different words. It serves as an excellent educational tool for students, researchers, and developers looking to gain a deeper insight into the foundational concepts of modern Natural Language Processing (NLP).

## Features

QuLab offers the following key functionalities:

*   **Self-Attention Visualization**:
    *   Interactively generate synthetic numerical sentences or input a custom text sentence.
    *   Visualize the self-attention weights within a single head using a heatmap.
    *   Understand how a single token attends to other tokens in the sequence.
    *   Parameters to control vocabulary size, maximum sentence length, and number of sentences for synthetic data.
*   **Multi-Head Attention Visualization**:
    *   Extend the self-attention concept to visualize multiple attention heads in parallel.
    *   Observe how different heads capture diverse relationships within the same input sequence.
    *   Interactive tabs to switch between attention heatmaps for each individual head.
    *   Adjust the number of attention heads.
*   **Positional Encoding Visualization**:
    *   Explore the mathematical basis of positional encodings.
    *   Visualize the sinusoidal positional encoding values as a heatmap across different positions and embedding dimensions.
    *   Understand how sequential information is injected into position-agnostic Transformer architecture.
    *   Adjust maximum sequence length and embedding dimension.
*   **Interactive Data Generation and Validation**:
    *   Built-in synthetic data generation for demonstration purposes.
    *   Robust data validation and exploration steps (missing values, data types, summary statistics) are displayed for the generated datasets, simulating real-world data science practices.
*   **Educational Context**: Each section includes detailed explanations of the business value, learning goals, and technical implementation of the respective Transformer component.

## Getting Started

Follow these instructions to set up and run the QuLab application on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)
*   `git` (for cloning the repository)

### Installation

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/QuLab.git
    cd QuLab
    ```
    *(Replace `your-username` with the actual GitHub username if this project is hosted.)*

2.  **Create a virtual environment** (recommended):

    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment**:

    *   **On Windows**:
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies**:
    Create a `requirements.txt` file in the root directory of your project with the following contents:

    ```
    streamlit
    torch
    numpy
    pandas
    transformers
    plotly
    ```

    Then install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Once the dependencies are installed, you can run the Streamlit application.

1.  **Run the application**:

    ```bash
    streamlit run app.py
    ```

    This command will open the application in your default web browser (usually at `http://localhost:8501`).

2.  **Navigation**:
    *   Use the sidebar on the left to navigate between the different visualization pages:
        *   `Self-Attention Visualization`
        *   `Multi-Head Attention`
        *   `Positional Encoding`

3.  **Interaction**:
    *   **Input Parameters**: On each page, adjust the sliders and text areas in the sidebar to modify parameters like `Vocabulary Size`, `Maximum Sentence Length`, `Number of Sentences` (for synthetic data), `Number of Attention Heads`, etc.
    *   **Custom Sentence Input**: For Self-Attention and Multi-Head Attention, you can enter your own sentence in the `Enter a sentence:` text area to analyze real text (tokenized using `bert-base-uncased`). If left empty, synthetic data will be used.
    *   **Run Analysis**: Click the "Run Analysis" or "Run MHA Analysis" or "Generate Positional Encoding" button to process the data and display the visualizations based on your chosen parameters.
    *   **Data Validation & Exploration**: Expand the "Show Data Validation Details" and "Show Summary Statistics" sections to inspect the underlying dataset generated or processed for the visualization.

## Project Structure

The project is organized into the following directories and files:

```
QuLab/
├── app.py                      # Main Streamlit application entry point
├── application_pages/          # Directory containing individual page logic
│   ├── page1.py                # Self-Attention Visualization logic
│   ├── page2.py                # Multi-Head Attention Visualization logic
│   └── page3.py                # Positional Encoding Visualization logic
├── requirements.txt            # List of Python dependencies
└── README.md                   # This README file
```

## Technology Stack

*   **Streamlit**: For creating the interactive web-based user interface.
*   **PyTorch**: Deep learning framework used for building the Self-Attention and Multi-Head Attention modules and tensor operations.
*   **Hugging Face Transformers**: Utilized for `AutoTokenizer` to process natural language input for attention visualizations.
*   **NumPy**: Fundamental package for scientific computing with Python, used for numerical operations.
*   **Pandas**: Data manipulation and analysis library, used for data validation and exploration.
*   **Plotly**: For generating interactive and visually appealing heatmaps for attention weights and positional encodings.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  **Fork** the repository.
2.  **Clone** your forked repository: `git clone https://github.com/your-username/QuLab.git`
3.  **Create a new branch**: `git checkout -b feature/your-feature-name` or `bugfix/your-bug-fix`
4.  **Make your changes** and commit them with descriptive messages.
5.  **Push** your branch to your forked repository.
6.  **Open a Pull Request** to the `main` branch of the original repository.

Please ensure your code adheres to good practices and includes appropriate documentation and tests (if applicable).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(Note: A `LICENSE` file would need to be created in the root directory)*

## Contact

For any questions or feedback, please reach out:

*   **Project Maintainer**: Quant University Team
*   **Email**: info@quantuniversity.com
*   **Website**: [www.quantuniversity.com](https://www.quantuniversity.com)