# PyTorch Transformer

This repository contains a clean, from-scratch implementation of a Transformer-based Model and a Byte Pair Encoding (BPE) tokenizer. It also includes custom Triton kernels implementing FlashAttention-2 for efficient training and inference. A demo model with 75,466,752 parameters was trained on the TinyStories dataset.

## Model architecture scheme

<p float="left">
  <img src="https://raw.githubusercontent.com/danilkos00/PyTorch-Detector/main/scheme/scheme.png" width="400"/>
</p>

## Features

- Transformer architecture implemented from scratch in PyTorch
- Custom BPE tokenizer
- FlashAttention-2 implemented with Triton
- TinyStories and OpenWebText datasets support and training scripts
- Colab demo for quick testing

## Quick Start

You can run the project either in Google Colab or locally.

### Launch via Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danilkos00/PyTorch-Transformer/blob/main/Transformer_demo.ipynb)

### Local Launch

1. Clone the repository:
   ```bash
   git clone https://github.com/danilkos00/PyTorch-Transformer.git
   cd PyTorch-Transformer

2. Install dependencies:
    ```bash
    pip install -r requirements.txt -qq

3. You can download pretrained weights from the following link:
    [Download pretrained model](https://drive.google.com/uc?id=1-L881Atoagz_0AXnwcKI28ZcWjq6NUy8)


## Demo-Model Details

- Parameter count: **75,466,752**
- Dataset: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- Attention: Supports both standard and FlashAttention-2

## FlashAttention-2 Implementation

Implemented via custom Triton kernels with the following characteristics:

- Tiled block-wise attention computation
- Numerically stable softmax and logsumexp
- Optimized for GPU performance and memory usage

## Tokenizer

A simple BPE tokenizer written from scratch:

- Learns merge rules from training data
- Supports encoding and decoding

## Project Structure
```
project_root/
├── config/             
│   └── config.json     # Configuration file for training and generation
├── data/               
│   └── dataset.py      # Script for loading dataset
├── src/               
│   ├── model.py            # Transformer model and attention modules
│   ├── optimizer.py        # AdamW and CosineWarmup scheduler implementation
│   ├── tokenizer.py        # BPE tokenizer implementation
│   ├── flash_attention.py  # FlashAttention-2 Triton kernels
│   ├── nn_tools.py         # Utility functions
│   ├── training.py         # Training functions
├── requirements.txt
└── README.md
```