# QuLab: Transformer Models Visualization

## Description
QuLab is an interactive Streamlit application designed to explore and visualize the self-attention mechanism of Transformer models. It provides users with the ability to understand how Transformers process sequences by weighing the importance of different words in a sentence. This application allows for the visualization of attention weights, generation of synthetic data, and a deeper insight into the "Attention Is All You Need" paradigm.

## Features
- **Self-Attention Visualization**: Visualize how self-attention works within a Transformer model.
- **Multi-Head Attention**: Explore the concept of multi-head attention and how different heads focus on various aspects of the input data.
- **Positional Encoding**: Understand the importance of positional encoding in Transformers and visualize the positional encodings.
- **Interactive Controls**: Adjust parameters such as vocabulary size, maximum sentence length, and number of sentences to generate synthetic data.
- **Data Validation and Exploration**: Validate and explore the generated synthetic data with summary statistics and dataframes.

## Getting Started

### Prerequisites
To run this application, you need to have Python installed along with the following libraries:
- Streamlit
- PyTorch
- Transformers
- NumPy
- Pandas
- Plotly

You can install the required libraries using pip:

```bash
pip install streamlit torch transformers numpy pandas plotly
```

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/qulab.git
   cd qulab
   ```
2. Install the required libraries as mentioned above.

## Usage
To run the application, navigate to the project directory and execute the following command:

```bash
streamlit run app.py
```

Once the application is running, you can access it in your web browser at `http://localhost:8501`.

### Basic Usage Instructions
- Use the sidebar to navigate between different pages: Self-Attention Visualization, Multi-Head Attention, and Positional Encoding.
- Input a sentence or adjust the parameters to generate synthetic data.
- Click on the "Run Analysis" button to visualize the attention weights or positional encodings.

## Project Structure
```
qulab/
│
├── app.py                  # Main application file
│
└── application_pages/      # Directory containing different application pages
    ├── page1.py            # Self-Attention Visualization
    ├── page2.py            # Multi-Head Attention Visualization
    └── page3.py            # Positional Encoding Visualization
```

## Technology Stack
- **Frontend**: Streamlit
- **Backend**: PyTorch
- **Data Processing**: NumPy, Pandas
- **Visualization**: Plotly
- **NLP**: Transformers (Hugging Face)

## Contributing
Contributions are welcome! If you would like to contribute to this project, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your forked repository.
5. Create a pull request describing your changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact
For any inquiries or feedback, please contact:
- **Your Name**: [your.email@example.com](mailto:your.email@example.com)
- **GitHub**: [yourusername](https://github.com/yourusername)

Feel free to reach out for any questions or suggestions regarding the QuLab application!
