# Comparison of First-Order stochastic Optimization Methods for Deep Learning Problems

## Introduction

This repository contains the implementation of first-order stochastic algorithms to train Deep Learning models: SGD, Momentum, NAG, ADG, RMS Prop, Adam, AMS Grad, Nadam, Adamax, Nostalgic Adam. We compare the performances of such algorithms in Deep Learning framework.

## Features
Python files where algorithms are implemented : 

* ```ModelImageClassification.py```: LeNet5 CNN and first-order stochastic algorithms for image classification.
* ```ModelRegression.py```: MLP and first-order stochastic algorithms for regression.
* ```ModelTextClassification.py```: LSTM and first-order stochastic algorithms for text classification.

Notebook files with experiments:

* ``` ExperimentImageClassification.ipynb ```: LeNet5 CNN used on MNIST Dataset.
* ``` ExperimentRegression.ipynb ```: MLP used on Boston Housing Dataset.
* ``` ExperimentTextClassification.ipynb ```: LSTM used on IMBD Dataset.
  
## Getting Started

### Prerequisites

- Python (>=3.6)
- Other dependencies (specified in `requirements.txt`)

### Installation

```bash
git clone https://github.com/louislat/Advanced-ML-Project
cd src
pip install -r requirements.txt
```
