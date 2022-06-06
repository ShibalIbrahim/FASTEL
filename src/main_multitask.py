import os
import pandas as pd, numpy as np, s3fs, matplotlib.pyplot as plt
import seaborn as sns
import data_utils
import timeit
import joblib
from copy import deepcopy
import seaborn as sb
import pandas as pd
import time
import pathlib
import argparse
import engine

parser = argparse.ArgumentParser(description='Multitask differentiable decision tree ensembles.')

# Data Arguments
parser.add_argument('--load_directory', dest='load_directory',  type=str, default='s3://cortex-mit1003-lmdl-workbucket/public-datasets-processed')
parser.add_argument('--data', dest='data',  type=str, default='atp1d')
parser.add_argument('--seed', dest='seed',  type=int, default=8)
parser.add_argument('--val_size', dest='val_size',  type=float, default=0.2)
parser.add_argument('--test_size', dest='test_size',  type=float, default=0.2)
parser.add_argument('--missing_percentage', dest='missing_percentage',  type=float, default=0.0)

# Model Arguments
parser.add_argument('--trees', dest='trees',  type=int, default=20)
parser.add_argument('--depth', dest='depth',  type=int, default=4)
parser.add_argument('--activation', dest='activation',  type=str, default='sigmoid')
parser.add_argument('--loss', dest='loss',  type=str, default='mse')
parser.add_argument('--architecture', default='shared', type=str) # only matters for ZIP, NB losses.
parser.add_argument('--model_type', default=None, type=str) # 'regularized' or None
parser.add_argument('--alpha', dest='alpha', type=float, default=0.1) # for regularization
parser.add_argument('--power', dest='power', type=float, default=1.0) # for regularization strength along depth

# Algorithm Arguments
parser.add_argument('--epochs', dest='epochs',  type=int, default=200)
parser.add_argument('--batch_size', dest='batch_size',  type=int, default=64)
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.01)
parser.add_argument('--patience', dest='patience',  type=int, default=25)

# Logging Arguments
parser.add_argument('--version', dest='version',  type=int, default=1)
parser.add_argument('--save_directory', dest='save_directory',  type=str, default='./runs/soft_trees/publicdata')

args = parser.parse_args()

print(args.data, args.load_directory)
df_X, df_y, metadata = data_utils.load_multitask_public_data(
    data=args.data,
    path=args.load_directory,
)
data_processed = data_utils.load_processed_multitask_public_data(
    df_X, df_y, metadata,
    'all',
    val_size=args.val_size,
    test_size=args.test_size,
    seed=args.seed,
    missing_percentage=args.missing_percentage,    
)


path = os.path.join(args.save_directory, args.data, "all-tasks", args.loss, "{}".format(args.version))
os.makedirs(path, exist_ok=True)

fastel = engine.MultiTaskTrees(
    data_processed.x_train_processed.shape[1:],
    loss_criteria=args.loss,
    architecture=args.architecture,
    activation=args.activation,
    num_trees=args.trees,
    depth=args.depth,
    num_tasks=data_processed.y_train_processed.shape[1],
    model_type=args.model_type,
    alpha=args.alpha,
    power=args.power,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    epochs=args.epochs,
)

fastel.train(
    data_processed.x_train_processed,
    data_processed.y_train_processed,
    data_processed.w_train,
    data_processed.x_valid_processed,
    data_processed.y_valid_processed,
    data_processed.w_valid
)

metrics_valid = fastel.evaluate(data_processed.x_valid_processed, data_processed.y_valid_processed, data_processed.w_valid)
metrics_test = fastel.evaluate(data_processed.x_test_processed, data_processed.y_test_processed, data_processed.w_test)
print("============Validation Metrics =================")
print(metrics_valid)
print("============Test Metrics =================")
print(metrics_test)