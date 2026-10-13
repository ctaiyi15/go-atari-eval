# Recognizing Atari and Assessing Sente: Controlled Evaluation Tasks for Language-Based Go Agents

This repository contains the datasets and code for the paper, *Recognizing Atari and Assessing Sente: Controlled Evaluation Tasks for Language-Based Go Agents*.


## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

A CUDA-compatible GPU is required for model training and inference.

## Dataset

The `data/` directory contains the training, validation, and test datasets.

Each training example consists of a Go board position, move history, candidate move, and target output containing Go terminology and associated annotations.

To obtain the dataset:
```bash
python scripts/preprocessing/download.py
python scripts/preprocessing/extract.py
python scripts/preprocessing/add_winrates.py
```

## Training

We fine-tune LoGos-7B using 4-bit QLoRA.

To train the model:

```bash
python scripts/train.py
```

The trained LoRA adapter is saved to `outputs/logo-qlora/final/`.

## Inference

To run the inference scripts on evaluation tasks:

```bash
python scripts/inf_sente.py
python scripts/inf_non_sente.py
```

Evaluation results are saved to the `outputs/` directory.

