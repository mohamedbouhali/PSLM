# PSLM - Psychological Support Language Model

A comprehensive framework for fine-tuning and evaluating language models on counseling and psychological support data using various prompting strategies.

## Overview

This repository contains a complete pipeline for:
- **Data Preparation**: Cleaning and structuring counseling chat data
- **Model Fine-tuning**: Training language models using LoRA on psychological support datasets
- **Evaluation**: Testing models with zero-shot, one-shot, and few-shot prompting strategies
- **Generation**: Producing therapeutic responses using fine-tuned models

## Project Structure

```
PSLM/
├── finetune/                    # Data preparation and training scripts
│   ├── CounselChatDataPrep.ipynb    # Main data cleaning notebook
│   ├── project.yaml                 # Training configuration
│   ├── system_prompt.txt            # System prompts for training
│   └── trained_inference.py         # Inference with fine-tuned models
├── finetuned-models/           # Trained model checkpoints
│   ├── lora-8-zero-shot/       # Zero-shot fine-tuned model
│   ├── lora-8-one-shot/        # One-shot fine-tuned model
│   └── lora-8-few-shot/        # Few-shot fine-tuned model
├── generate-zero-single-few-shot.py  # Main evaluation script
├── generate2.py                 # Alternative generation script
└── zero-single-few-shots-config.yaml # Configuration for evaluation
```

## Data Preparation

### Input Data Format
The system processes counseling data with the following columns:
- `questionID`, `questionTitle`, `questionText`, `questionLink`
- `topic`, `therapistInfo`, `therapistURL`
- `answerText`, `upvotes`, `views`

### Data Processing Pipeline
1. **Cleaning**: Remove rows with missing essential fields
2. **Filtering**: Select top-scoring entries based on `upvotes + views`
3. **Output Generation**: Create training datasets in JSONL format

### Generated Datasets
- **`qa_train.jsonl`**: Q&A instruction-tuning dataset
- **`classification_train.jsonl`**: Topic classification dataset

## Training

### Quick Start
Use the MS-Swift framework for fine-tuning:

```bash
swift sft \
  --model LLM-Research/gemma-3-1b-it \
  --dataset ./finetune/qa_train.jsonl ./finetune/classification_train.jsonl \
  --train_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --gradient_accumulation_steps 16 \
  --eval_steps 50 \
  --save_steps 50 \
  --save_total_limit 2 \
  --logging_steps 5 \
  --max_length 2048 \
  --output_dir output
```

### Training Configuration
- **Base Model**: Google Gemma models (2B/3B variants)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Strategy**: Multiple approaches (zero-shot, one-shot, few-shot)

## Evaluation & Generation

### Prompting Strategies
- **Zero-shot**: Direct question answering without examples
- **One-shot**: Single example demonstration
- **Few-shot**: Multiple examples for context

### Running Evaluation
```bash
python generate-zero-single-few-shot.py
```

### Configuration
Modify `zero-single-few-shots-config.yaml` to adjust:
- Prompting strategy
- Number of examples
- System prompts
- Output formatting

## Model Performance

The repository includes pre-trained models with different approaches:
- **LoRA Rank 8**: Optimized for efficiency
- **Multiple Strategies**: Compare different prompting methods
- **Training Metrics**: Comprehensive evaluation logs and visualizations

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- MS-Swift (for training)
- CUDA-compatible GPU (recommended)

## Notes

- Monitor GPU usage during training: `sudo powermetrics --samplers gpu_power -i 1000 -n 10`
- Adjust the `n` parameter in `CounselChatDataPrep.ipynb` to control data filtering
- All fine-tuned models are saved with comprehensive training logs and metrics

## Contributing

This project focuses on psychological support and counseling applications. Please ensure all contributions maintain appropriate ethical standards for mental health applications.