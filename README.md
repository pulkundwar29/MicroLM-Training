# NLP Model Training Project

This repository contains implementations of different natural language processing models, including a GPT-like language model and a fine-tuned BERT model for Natural Language Inference (NLI) tasks.

## Project Structure

### Features
- **GPT-like language model implementation** using PyTorch
  - Custom token embedding
  - Text generation capabilities
  - Training and validation split functionality
- **BERT model fine-tuning for NLI tasks**
  - Uses the Sentence Transformers library
  - Multiple Negatives Ranking Loss
  - Evaluation on STS Benchmark dataset
- **Integration with Hugging Face Hub**

## Requirements
- Python 3.x
- PyTorch
- Sentence Transformers
- Datasets (Hugging Face)
- CUDA-capable GPU (optional, but recommended)

## Setup
1. Clone this repository.
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```
## Usage

### GPT Language Model
Run the Jupyter notebook `gpt.ipynb` to train and experiment with the GPT-like language model. The model is configured with:
- **Block size**: 8
- **Batch size**: 4
- **Learning rate**: 3e-3
- **Training iterations**: 1000

### BERT Fine-tuning
Run the NLI training script:

```bash
python train_nli.py
```
The default model is `"distilroberta-base"` if no model name is provided.

### Model Training Parameters
- **Training batch size**: 128
- **Maximum sequence length**: 75
- **Number of epochs**: 1
- **FP16 training**: Enabled
- **Evaluation steps**: Every 10 steps
- **Warmup ratio**: 0.1

