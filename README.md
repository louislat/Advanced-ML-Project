# Comparison of First-Order stochastic Optimization Methods for Deep Learning Problems

## Introduction

This repository contains the implementation of first-order stochastic algorithms to train Deep Learning models: SGD, Momentum, NAG, ADG, RMS Prop, AdaDelta, Adam, AMS Grad, Nadam, Adamax, Nostalgic Adam. We compare the performances of such algorithms in Deep Learning framework.

## Features
Python files where algorithms are implemented : 

* ```ModelImageClassification.py```: LeNet5 CNN and first-order stochastic algorithms for image classification.
* ```ModelNostalgic.py```: LeNet5 CNN and first-order stochastic algorithms for image classification. Focus on Nostalgic Adam algorithm.
* ```ModelRegression.py```: MLP and first-order stochastic algorithms for regression.
* ```ModelTextClassification.py```: MLP and first-order stochastic algorithms for text classification.

Notebook files with experiments:

* ``` ExperimentImageClassification.ipynb ```: LeNet5 CNN used on MNIST Dataset.
* ``` ExperimentRegression.ipynb ```: MLP used on Boston Housing Dataset.
* ``` ExperimentTextClassification.ipynb ```: MLP used on IMBD Dataset.
* ``` ExperimentNostalgicAdam.ipynb ```: LeNet5 CNN used on MNIST Dataset with Nostalgic Adam algorithm.
  
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
