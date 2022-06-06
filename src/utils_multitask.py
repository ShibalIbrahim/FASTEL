"""Evaluation metrics for multitask learning
"""
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn import metrics

def multitask_masked_mean_squared_error(y_true, y_pred, sample_weight):
    """Computes mean squared error for multitask with missing and 0 responses.
    
    Args:
        y_true: True responses, a numpy array of shape (num_samples, num_tasks). 
        y_pred: Predicted responses, a numpy array of shape (num_samples, num_tasks).
        sample_weight: Sample weights, a numpy array of shape (num_samples, ).
    
    Returns:
        weighted normalized poisson deviance, float scalar.
        - Note normalization is with respect to each task to correct for number of available responses for each task.
    """
    mask = ~np.isfinite(y_true) # prevents inclusion of NaNs in responses
    y_true = np.ma.masked_array(y_true, mask=mask)
    y_pred = np.ma.masked_array(y_pred, mask=mask)
    if len(y_true.shape)>1:
        sample_weight = np.tile(sample_weight, (y_true.shape[1], 1)).transpose()
    sample_weight = np.ma.masked_array(sample_weight, mask=mask)
    
    diff = (np.square(y_true - y_pred)) * sample_weight
    return (np.sum(diff, axis=0) / np.sum(sample_weight, axis=0)).data


def multitask_masked_poisson_deviance(y_true, y_pred, sample_weight, eps=1e-6):
    """Computes poisson deviance for multitask with missing and 0 responses.
    
    Args:
        y_true: True responses, a numpy array of shape (num_samples, num_tasks). 
        y_pred: Predicted responses, a numpy array of shape (num_samples, num_tasks).
        sample_weight: Sample weights, a numpy array of shape (num_samples, ).
    
    Returns:
        weighted normalized poisson deviance, float scalar.
        - Note normalization is with respect to each task to correct for number of available responses for each task.
    """
    y_pred[y_pred<0.0] = 0.0
    mask = ~np.isfinite(y_true) # prevents inclusion of NaNs in responses
    y_true = np.ma.masked_array(y_true, mask=mask)
    y_pred = np.ma.masked_array(y_pred, mask=mask)
    if len(y_true.shape)>1:
        sample_weight = np.tile(sample_weight, (y_true.shape[1], 1)).transpose()
    sample_weight = np.ma.masked_array(sample_weight, mask=mask)
    
    # Compute weighted deviance for zero y_true entries
    y_true_zeros = np.ma.masked_where(y_true != 0, y_true)
    dev_per_task_case_zeros = np.sum((-y_true_zeros + y_pred) * sample_weight, axis=0)
    dev_per_task_case_zeros = dev_per_task_case_zeros.filled(fill_value=0.0)
    
    # Compute weighted deviance for nonzero y_true entries
    y_true_nonzeros = np.ma.masked_where(y_true==0, y_true)
    dev_per_task_case_nonzeros = np.sum((y_true_nonzeros * np.log((y_true_nonzeros) / (y_pred + eps)) - y_true_nonzeros + y_pred) * sample_weight, axis=0)
    dev_per_task_case_nonzeros = dev_per_task_case_nonzeros.filled(fill_value=0.0)
    
    # Add both deviances and compute weighted normalized deviance
    dev_per_task = dev_per_task_case_zeros + dev_per_task_case_nonzeros
    poisson_dev = dev_per_task / np.sum(sample_weight, axis=0)
    
    poisson_dev_zeros = dev_per_task_case_zeros / np.sum(sample_weight, axis=0)
    poisson_dev_nonzeros = dev_per_task_case_nonzeros / np.sum(sample_weight, axis=0)
    return poisson_dev.data, poisson_dev_zeros.data, poisson_dev_nonzeros.data