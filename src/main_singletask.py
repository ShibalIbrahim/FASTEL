import os
import pandas as pd, numpy as np, s3fs, matplotlib.pyplot as plt
import seaborn as sns
from utilities import modeling_data, create_data_for_lgbm, poisson_deviance
import data_utils
import optuna
from optuna.samplers import RandomSampler
import timeit
import joblib
from copy import deepcopy
import seaborn as sb
import pandas as pd
import time
import pathlib
import argparse
import singletask_tuning

from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']

parser = argparse.ArgumentParser(description='Singletask learning with Tree Ensembles.')

# Data Arguments
parser.add_argument('--load_directory', dest='load_directory',  type=str, default='s3://cortex-mit1003-lmdl-workbucket/public-datasets-processed')
parser.add_argument('--data', dest='data',  type=str, default='atp1d')
parser.add_argument('--seed', dest='seed',  type=int, default=8)
parser.add_argument('--val_size', dest='val_size',  type=float, default=0.2)
parser.add_argument('--test_size', dest='test_size',  type=float, default=0.2)
parser.add_argument('--missing_percentage', dest='missing_percentage',  type=float, default=0.0)

# Model Arguments
parser.add_argument('--architecture', default='shared', type=str) # only matters for zero-inflation models

# Algorithm Arguments
parser.add_argument('--task', dest='task',  type=int, default=0)
parser.add_argument('--loss', dest='loss',  type=str, default='mse')
parser.add_argument('--max_trees', dest='max_trees',  type=int, default=100)
parser.add_argument('--max_depth', dest='max_depth',  type=int, default=4)
parser.add_argument('--activation', dest='activation',  type=str, default='sigmoid')
parser.add_argument('--max_epochs', dest='max_epochs',  type=int, default=200)
parser.add_argument('--n_trials', dest='n_trials',  type=int, default=2)
parser.add_argument('--patience', dest='patience',  type=int, default=25)

# Logging Arguments
parser.add_argument('--version', dest='version',  type=int, default=1)
parser.add_argument('--save_directory', dest='save_directory',  type=str, default='./logs/soft_trees/publicdata')

# Tuning Arguments
parser.add_argument('--tuning_criteria', dest='tuning_criteria',  type=str, default='negloglikelihood')
parser.add_argument('--tuning_seed', dest='tuning_seed',  type=int, default=1)

args = parser.parse_args()

print("Load Directory:", args.load_directory)
print("Data:", args.data)
print("Task:", args.task)
df_X, df_y, metadata = data_utils.load_multitask_public_data(
    data=args.data,
    path=args.load_directory,
)
data_processed = data_utils.load_processed_multitask_public_data(
    df_X, df_y, metadata,
    args.task,
    val_size=args.val_size,
    test_size=args.test_size,
    seed=args.seed,
    missing_percentage=args.missing_percentage,
)


path = os.path.join(args.save_directory, args.data, "task-{}".format(args.task), args.loss, "{}.{}".format(args.version, args.tuning_seed))
os.makedirs(path, exist_ok=True)

