# Text Classification for Web of Science

This project is designed to classify scientific texts using state-of-the-art machine learning models. 
It includes various deep learning models like BERT and a custom CNN architecture specifically tailored for handling multi-class classification problems in text data.


## Project Structure

- `data_loader/`: Contains scripts for loading and preprocessing the dataset.
- `models/`: Includes the model definitions used for text classification.
  - `bert/`: Contains the BERT classifier implementation.
    - `vector_cache/`: Stores precomputed vectors for faster processing.
    - `output_dir/`: The directory where trained model outputs and checkpoints are saved.
    - `bert_classifier.py`: The main script for the BERT model training and prediction.
  - `HDLT_CNN/`: Houses the custom CNN architecture for hierarchical text classification.

## Getting Started

To set up this project, please follow the instructions below.

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- PyTorch
- Transformers library

You can install the required libraries using the `requirements.txt` file:
```bash
pip install -r -U requirements.txt
