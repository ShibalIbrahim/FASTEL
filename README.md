# FASTEL

Shibal Ibrahim, Hussein Hazimeh, and Rahul Mazumder

Massachusetts Institute of Technology

## Introduction

FASTEL (Flexible and Scalable Tree Ensemble Learning) is a flexible and scalable toolkit for learning tree ensembles with seamless support for new loss functions. We introduce a novel, tensor-based formulation for differentiable tree ensembles that allows for efficient training on GPUs.  We extend differentiable tree ensembles to multi-task learning settings by introducing a new regularizer that allows for soft parameter sharing across tasks. Our framework can lead to 100x more compact ensembles and up to 23% improvement in out-of-sample performance, compared to tree ensembles learnt by popular toolkits such as XGBoost. See [Flexible Modeling and Multitask Learning using Differentiable
Tree Ensembles](https://arxiv.org/abs/2205.09717) for details.

## Installation
FASTEL is written in Tensorflow 2.4. It uses Tensorflow-Probability (0.12) internally (for flexibility in modeling to support zero-inflation, negative binomial regression loss functions). Before installing FASTEL, please make sure that Tensorflow 2 and Tensorflow-Probability are installed.


## Citing FASTEL
If you find this work useful in your research, please consider citing the following paper:

```
@article{Ibrahim2022,
    title={Flexible Modeling and Multitask Learning using Differentiable Tree Ensembles},    
    author={Shibal Ibrahim, Hussein Hazimeh and Rahul Mazumder},
    year={2022},
    eprint={2205.09717},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
