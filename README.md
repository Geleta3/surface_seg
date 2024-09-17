# Pixel-level Surface Defect Detection for Magnetic Tile Images

This repository provides a robust framework for pixel-level surface defect detection, focusing on Magnetic Tile images. It offers a comprehensive solution, from dataset loading and model training to evaluation, leveraging advanced computer vision techniques and deep learning algorithms.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)

## Introduction

Pixel-level surface defect detection plays a crucial role in ensuring product quality across industries. This repository addresses the challenge of identifying and classifying surface defects on Magnetic Tile images, a task vital for maintaining high manufacturing standards.

## Dataset

The repository includes a dedicated dataset module (`dataset`) designed to handle the Magnetic Tile defect dataset. It provides functionalities for:

- Loading and preprocessing images and corresponding masks.
- Splitting the dataset into training and validation sets.
- Supporting data augmentation techniques to enhance model robustness.

## Model

The `models` module provides implementations of various deep learning architectures suitable for semantic segmentation, the core task in pixel-level defect detection. The repository currently supports:

- **U-Net:** A widely adopted architecture for biomedical image segmentation, easily adaptable for defect detection.

The modular design allows for easy integration of additional architectures.

## Training

The training process is streamlined with a dedicated training script (`train.py`). Key features include:

- **Configuration:** A centralized configuration file (`config.py`) allows for easy modification of hyperparameters, dataset paths, and model choices.
- **Loss Function:** The `utils.loss` module provides implementations of commonly used loss functions for segmentation tasks, such as cross-entropy loss.
- **Optimizer and Scheduler:** The training script utilizes the Adam optimizer and a learning rate scheduler to optimize the training process.
- **Logging and Checkpointing:** The `utils.logger` module enables logging of training metrics and saving model checkpoints for later analysis or resumption.

## Evaluation

The `evaluate.py` script facilitates a comprehensive evaluation of the trained model. It calculates metrics such as:

- **IOU (Intersection over Union):** A standard metric for evaluating segmentation performance.
- **F1-score:**  A measure of the model's accuracy considering both precision and recall.
- **Average Precision (AP):** A metric that provides a comprehensive assessment of the model's performance across different confidence thresholds.

The evaluation script also visualizes predictions and saves them for qualitative analysis.

## Repository Structure
└──
├── dataset
│ ├── pycache
│ │ └── ...
│ ├── dataset.py
│ ├── aitex.py
│ ├── dagm.py
│ ├── magnetic.py
│ ├── mt.py
│ └── neu.py
├── figure
│ └── MT.jpg
├── models
│ ├── pycache
│ │ └── ...
│ ├── init.py
│ ├── model.py
│ ├── pretrained.py
│ └── unet.py
└── utils
├── pycache
│ └── ...
├── logger.py
├── loss.py
├── utils.py
└── visualization.py
├── pycache
│ └── ...
├── init.py
├── README.md
├── config.py
├── evaluate.py
└── train.py
- **`dataset/`:** Contains dataset loading and preprocessing code.
- **`models/`:**  Houses the implementations of different deep learning models.
- **`utils/`:**  Includes utility functions for loss calculation, logging, and visualization.
- **`config.py`:**  Centralized configuration file.
- **`train.py`:**  Script for training the model.
- **`evaluate.py`:**  Script for evaluating the trained model.

## Getting Started

1. **Clone the repository:** `git clone https://github.com/your-username/your-repository.git`
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Prepare the dataset:** Download and structure the Magnetic Tile defect dataset according to the instructions in the `dataset/` directory.
4. **Configure the training:** Modify the `config.py` file to adjust hyperparameters, dataset paths, and model choices.
5. **Train the model:** `python train.py`
6. **Evaluate the model:** `python evaluate.py`
