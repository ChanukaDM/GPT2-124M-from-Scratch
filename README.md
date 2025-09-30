# GPT-2 124M Training from Scratch

Implementation of GPT-2 (124M parameters) trained from scratch on the FineWeb-Edu 10B dataset. This implementation features optimized training with Distributed Data Parallel (DDP), Flash Attention, and modern training techniques.

## Features

- **Model Architecture**: Full GPT-2 implementation with 12 layers, 12 attention heads, 768 embedding dimensions
- **Optimized Training**: 
  - Flash Attention for efficient attention computation
  - Distributed Data Parallel (DDP) for multi-GPU training
  - Mixed precision training (bfloat16)
  - Gradient accumulation for large effective batch sizes
  - Cosine learning rate scheduling with warmup
- **Dataset**: FineWeb-Edu 10B tokens (100 shards × 100M tokens each)
- **Evaluation**: HellaSwag benchmark (~25% accuracy)

## Model Configuration

```python
GPTConfig:
    block_size: 1024      # Maximum sequence length
    vocab_size: 50257     # GPT-2 BPE vocabulary
    n_layer: 12           # Number of transformer blocks
    n_head: 12            # Number of attention heads
    n_embd: 768           # Embedding dimensions
    epochs: 19073         # Total training steps
```

## Training Specifications

### Hyperparameters

- **Total Batch Size**: 491,520 tokens (~0.5M tokens)
- **Micro Batch Size**: 4 sequences
- **Sequence Length**: 1,024 tokens
- **Learning Rate**: 
  - Max: 6e-4
  - Min: 6e-5 (10% of max)
  - Warmup Steps: 715
  - Schedule: Cosine decay
- **Optimizer**: AdamW
  - Betas: (0.9, 0.96)
  - Weight Decay: 0.1
  - Epsilon: 1e-8
- **Gradient Clipping**: Max norm 1.0

### Performance Optimizations

- TensorFloat32 (TF32) matrix multiplication
- Fused AdamW optimizer
- Mixed precision training (bfloat16)
- Flash Attention (PyTorch SDPA)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gpt2-training
cd gpt2-training

# Install dependencies
pip install tiktoken numpy transformers
```

## Dataset Preparation

The training script expects the FineWeb-Edu 10B dataset in the following structure:

```
edu_fineweb10B/
├── train_000.npy
├── train_001.npy
├── ...
├── train_099.npy
├── val_000.npy
└── val_001.npy
```

Each `.npy` file contains approximately 100M tokens encoded as numpy arrays.

## Training

#### Single GPU Training

```bash
python train.py
```
#### Multi-GPU Training (DDP)

```bash
# For 4 GPUs
torchrun --standalone --nproc_per_node=4 train.py
```


## Tokenizer

This implementation uses the GPT-2 tokenizer via `tiktoken`:

```python
import tiktoken
tokenizer = tiktoken.get_encoding('gpt2')
tokens = tokenizer.encode("Your text here")
```

### HellaSwag Benchmark

The model is evaluated on HellaSwag every 200 steps:
- **Metric**: Normalized accuracy
- **Expected Performance**: ~25% accuracy during training
- **Evaluation Method**: Zero-shot, selecting the most likely continuation

### Sample Generation

Every 50 steps, the model generates sample text to monitor training progress:

```python
Prompt: "Hello this is AI model,"
# Model generates 4 samples with top-k sampling (k=50)
```


## Loading Pretrained GPT-2

The implementation supports loading pretrained GPT-2 weights from HuggingFace:

```python
model = GPT.from_pretrained('gpt2')  # 124M parameters
# Also supports: gpt2-medium, gpt2-large, gpt2-xl
```

### Structure

```
.
├── train.py              # Main training script
├── hellaswag.py          # HellaSwag evaluation utilities
├── edu_fineweb10B/       # Training data directory
├── log/                  # Logs and checkpoints
└── README.md             # This file
```

## Key Implementation Details

### Architecture Highlights

1. **Causal Self-Attention**: Uses Flash Attention via `F.scaled_dot_product_attention`
2. **Weight Tying**: Token embeddings are tied with the output projection layer
3. **Layer Normalization**: Pre-normalization (before attention and MLP)
4. **Activation**: GELU with tanh approximation
5. **Residual Connections**: Clean residual paths throughout

### Training Features

1. **Gradient Accumulation**: Enables large effective batch sizes without OOM
2. **DDP**: Data-parallel training across multiple GPUs
3. **Learning Rate Schedule**: Linear warmup → cosine decay
4. **Weight Decay**: Applied to 2D parameters (weights), not to biases/LayerNorm
5. **Mixed Precision**: bfloat16 for forward/backward, float32 for optimizer


### References

- [NanoGPT Repository](https://github.com/karpathy/nanoGPT.git)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [FineWeb Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb)


## Acknowledgments

- Inspired by Andrej Karpathy's "Lets reproduce GPT2 124M"  and nanoGPT repo
- such a wonderful teacher : )


