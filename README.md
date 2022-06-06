# FASTEL

Shibal Ibrahim, Hussein Hazimeh, and Rahul Mazumder

Massachusetts Institute of Technology

## Introduction

FASTEL (Flexible and Scalable Tree Ensemble Learning) is a flexible and scalable toolkit for learning tree ensembles. We introduce a novel, tensor-based formulation for differentiable tree ensembles that allows for efficient training on GPUs.  We extend differentiable tree ensembles to multi-task learning settings by introducing a new regularizer that allows for soft parameter sharing across tasks. Our framework can lead to 100x more compact ensembles and up to 23% improvement in out-of-sample performance, compared to tree ensembles learnt by popular toolkits such as XGBoost. See our paper [Flexible Modeling and Multitask Learning using Differentiable
Tree Ensembles](https://arxiv.org/abs/2205.09717) appearing in 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining 2022 for details.

## Installation
FASTEL is written in Tensorflow 2.4. It uses Tensorflow-Probability (0.12) internally (for flexibility in modeling to support zero-inflation, negative binomial regression loss functions). Before installing FASTEL, please make sure that Tensorflow-GPU 2 and Tensorflow-Probability are installed.

## Support
The toolkit supports the following features:
1. singletask and multitask regression
2. missing responses
3. zero-inflated-poisson
4. negative binomial regression
5. GPU training


## Example Usage
```
import engine
input_shape = x_train.shape[1:]

# Define the Mutlitask Tree Ensemble model: here we choose 20 trees, each of depth 3.
# num_tasks is the number of regression targets.
# architecture='shared' corresponds to common splits for:
# 1. modeling mean and mixture components in zero-inflated model
# 2. modeling mean and dispersion components in negative binomial
# model_type='regularized' corresponds to enforcing soft sharing by imposing closeness in split hyperparameters across tasks, the strength
# is set by alpha parameter.
# model_type=None enforces hard sharing of split hyperparameters across tasks.
fastel = engine.MultiTaskTrees(
    input_shape,
    loss_criteria='zero-inflated-poisson',
    architecture='shared',
    activation='sigmoid',
    num_trees=20,
    depth=2,
    num_tasks=3,
    model_type='regularized',
    alpha=0.1,
    power=1.0,
    batch_size=64,
    learning_rate=0.01,
    epochs=200,
)

fastel.train(
    x_train, y_train, w_train,
    x_valid, y_valid, w_valid, 
)

metrics_valid = fastel.evaluate(x_valid, y_valid, w_valid)
metrics_test = fastel.evaluate(x_test, y_test, w_test)
print("============Validation Metrics =================")
print(metrics_valid)
print("============Test Metrics =================")
print(metrics_test)
```

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
