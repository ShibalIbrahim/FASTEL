"""Evaluation metrics for multitask learning
"""
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr 
# import seaborn as sns
from sklearn import metrics
from sklearn.feature_selection import mutual_info_regression

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

def multitask_masked_poisson_deviance_thresholds(y_true, y_pred, sample_weight):
    """Computes scaled poisson deviance for multitask with missing and 0 responses.
    
    Args:
        y_true: True responses, a numpy array of shape (num_samples, num_tasks). 
        y_pred: Predicted responses, a numpy array of shape (num_samples, num_tasks).
        sample_weight: Sample weights, a numpy array of shape (num_samples, ).
    
    Returns:
        weighted normalized poisson deviance, float scalar.
        - Note normalization is with respect to each task to correct for number of available responses for each task.
    """
    _, num_tasks = y_true.shape
    thresholds = np.max(y_pred, axis=0)+1e-6
    y_trues = np.array_split(y_true, num_tasks, axis=1)
    y_preds = np.array_split(y_pred, num_tasks, axis=1)
    scalers = np.ones(num_tasks)
    scaled_poisson_devs, _, _ = multitask_masked_poisson_deviance(y_true, y_pred, sample_weight)
    percentiles = 105*np.ones(num_tasks)
    for i, (y_t, y_p) in enumerate(zip(y_trues, y_preds)):
        scalers[i] = np.sum(np.mean(y_t[~np.isnan(y_t)], axis=0)/np.mean(y_p[~np.isnan(y_t)], axis=0))
        for per in np.linspace(start=100, stop=80, num=100, endpoint=True):
            yt = deepcopy(y_t)
            yp = deepcopy(y_p)
            thresh = np.percentile(yp.reshape(-1), per, axis=0)
#             print("Task: ", i,
#                   "per:", per,
#                   "threshold:", thresh,
#                   "scaler:", scalers[i])
            
            yp[yp>thresh] *= scalers[i]
            scaled_poisson_dev, scaled_poisson_dev_zeros, scaled_poisson_dev_nonzeros = multitask_masked_poisson_deviance(yt, yp, sample_weight)
            scaled_poisson_dev = np.sum(scaled_poisson_dev)            

            if scaled_poisson_dev < scaled_poisson_devs[i]:
                scaled_poisson_devs[i] = deepcopy(scaled_poisson_dev)
                thresholds[i] = deepcopy(thresh)
                percentiles[i] = deepcopy(per)
#                 print("scaled_poisson_devs", scaled_poisson_devs)
        y_pred[y_pred[:,i]>thresholds[i],i] *= scalers[i]
    
    return thresholds, scalers, percentiles, y_pred

def multitask_masked_convert_to_binary_labels(y_true):
    """
    Args:
        y_true: True responses, a numpy array of shape (num_samples, num_tasks). 
        
    Returns:
        y_classes: Binary targets,  a numpy array of shape (num_samples, num_tasks).
    """
    _, num_tasks = y_true.shape
    y_true_classes = np.nan*np.ones_like(y_true)
    for i in range(num_tasks):
        y_true_classes[~np.isnan(y_true[:,i]),i] = y_true[~np.isnan(y_true[:,i]),i]!=0.0
    return y_true_classes

def multitask_masked_convert_to_binary_prediction(y_prob, thresholds):
    """
    Args:
        y_prob: Predicted probabilities, a numpy array of shape (num_samples, num_tasks). 
        thresholds: threshold for classification, a numpy array of shape (num_tasks,). 
        
    Returns:
        y_classes: Binary predicted labels,  a numpy array of shape (num_samples, num_tasks).
    """
    _, num_tasks = y_prob.shape
    y_pred_classes = np.zeros_like(y_prob)
    for i in range(num_tasks):
        y_pred_classes[:,i] = y_prob[:,i]>thresholds[i]
    return y_pred_classes

def multitask_masked_roc_auc(y_true, y_prob, sample_weight):
    """Computes weighted roc auc for multitask with missing and 0 responses.
    
    Args:
        y_true: True labels, a numpy array of shape (num_samples, num_tasks). 
        y_prob: Predicted probabilities, a numpy array of shape (num_samples, num_tasks).
        sample_weight: Sample weights, a numpy array of shape (num_samples, ).
    
    Returns:
        roc auc, float scalar.
        best threshold for roc auc, float scalar.
        weighted roc auc, float scalar.
        best threshold for weighted roc auc, float scalar.
    """               
    _, num_tasks = y_true.shape
    roc_aucs = []
    roc_auc_best_thresholds = []
    for yt, yp in zip(np.array_split(y_true, num_tasks, axis=1),
                      np.array_split(y_prob, num_tasks, axis=1)):
        yt = yt.reshape(-1)
        yp = yp.reshape(-1)
        sw = sample_weight.reshape(-1)
        sw = sw[~np.isnan(yt)]
        yp = yp[~np.isnan(yt)]
        yt = yt[~np.isnan(yt)]
        
        # Sensitivity = True Positive Rate,   Specificity = 1 – False Positive Rate
        # Youden's J statistic: Sensitivity + Specificity – 1 = TruePositiveRate – FalsePositiveRate
        
        # ROC AUC
        fpr, tpr, thresholds = metrics.roc_curve(yt, yp, sample_weight=sw, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        roc_aucs.append(auc)
        J = tpr - fpr 
        ix = np.argmax(J)
        best_thresh = thresholds[ix]
        roc_auc_best_thresholds.append(best_thresh)
               
    return roc_aucs

def max_accuracy_from_roc_curve(y_true, y_pred, sample_weight):
    """Computes balanced accuracy and accuracy with missing responses.
    Args:
        y_true: True responses, a float numpy array of shape (num_samples, ).
        y_pred: Predicted responses, a float numpy array of shape (num_samples, ).
        sample_weight: Sample weights, a float numpy array of shape (num_samples, ).
    Returns:
        max_acc - best accuracy for one task
        acc_threshold - best accuracy threshold for one task
        max_balanced_accuracy - best balanced accuracy for one task
        balanced_accuracy_threshold - best balanced accuracy threshold for one task
    """
    #take care of NANs
    sample_weight = sample_weight[~np.isnan(y_true)]
    y_pred = y_pred[~np.isnan(y_true)]
    y_true = y_true[~np.isnan(y_true)]
    y_true  = np.asarray(y_true,  dtype=np.bool_)
    y_pred = np.asarray(y_pred, dtype=np.float_)
    #nlog(n) sorting
    sort_ordering = np.argsort(y_pred)
    #sort true label and sample weights
    y_true = y_true[sort_ordering]
    sample_weight = sample_weight[sort_ordering]
    #identify thresholds - for simplicity lets take them from actual values of y_pred
    thresholds = np.insert(y_pred[sort_ordering],0,0)
    len(thresholds)
    #now lets look at all parts of classification metrics that we actually care
    #initiate them as:
    TP = [sum(y_true*sample_weight)]
    TN = [0]
    FP = [sum(~y_true*sample_weight)]
    FN = [0]
    #Now we iterate over all thresholds
    for i in range(1, thresholds.size):
        TP.append(TP[-1] - int(y_true[i-1])*sample_weight[i-1])
        TN.append(TN[-1] + int(~y_true[i-1])*sample_weight[i-1])
        FP.append(FP[-1] - int(~y_true[i-1])*sample_weight[i-1])
        FN.append(FN[-1] + int(y_true[i-1])*sample_weight[i-1])
    TP = np.asarray(TP, dtype=np.float_)
    FP = np.asarray(FP, dtype=np.float_)
    TN = np.asarray(TN, dtype=np.float_)
    FN = np.asarray(FN, dtype=np.float_)
    #compute all accuracies
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    TPR = (TP) / (TP+FN)
    TNR = (TN) / (TN+FP)
    balanced_accuracy = (TPR+TNR)/2
    max_acc = max(accuracy)
    acc_threshold = thresholds[np.argmax(accuracy)]
    max_balanced_accuracy = max(balanced_accuracy)
    balanced_accuracy_threshold = thresholds[np.argmax(balanced_accuracy)]
    return (max_acc, acc_threshold, max_balanced_accuracy, balanced_accuracy_threshold)

def multitask_masked_accuracy(y_true, y_pred, sample_weight, acc_thresholds=None, bal_acc_thresholds=None):
    """Iterator over all tasks to compute number
    Args:
        y_true: True responses, a float numpy array of shape (num_tasks, num_samples).
        y_pred: Predicted responses, a float numpy array of shape (num_tasks, num_samples).
        sample_weight: Sample weights, a float numpy array of shape (num_samples, ).
    Returns:
        accs: accuracy, a float numpy array of shape (num_tasks,).
        accs_thresholds: accuracy thresholds, a float numpy array of shape (num_tasks,).
        balanced_accs: balanced accuracy, a float numpy array of shape (num_tasks,).
        balanced_accs_thresholds: balanced accuracy thresholds, a float numpy array of shape (num_tasks,).
    """
    num_tasks = y_pred.shape[1]
    accs = []
    accs_thresholds = []
    balanced_accs = []
    balanced_accs_thresholds = []
    for i in range(num_tasks):
        if acc_thresholds is None and bal_acc_thresholds is None:
            acc, thresh_acc, bal_acc, thresh_bal_acc = max_accuracy_from_roc_curve(y_true[:,i],
                                                                                   y_pred[:,i],
                                                                                   sample_weight)
        else:
            yt = deepcopy(y_true[:,i])
            yp = deepcopy(y_pred[:,i])
            sw = deepcopy(sample_weight)
            
            yt = yt.reshape(-1)
            yp = yp.reshape(-1)
            sw = sw.reshape(-1)
            
            sw = sw[~np.isnan(yt)]
            yp = yp[~np.isnan(yt)]
            yt = yt[~np.isnan(yt)]
            acc = metrics.accuracy_score(yt, yp>acc_thresholds[i], sample_weight=sw)
            thresh_acc = acc_thresholds[i]
            bal_acc = metrics.balanced_accuracy_score(yt, yp>bal_acc_thresholds[i], sample_weight=sw)
            thresh_bal_acc = bal_acc_thresholds[i]
            
        accs.append(acc)
        accs_thresholds.append(thresh_acc)
        balanced_accs.append(bal_acc)
        balanced_accs_thresholds.append(thresh_bal_acc)
    return accs, accs_thresholds, balanced_accs, balanced_accs_thresholds

# def multitask_masked_accuracy(y_true, y_pred, sample_weight):
#     """Computes balanced accuracy with missing responses.
#     Args:
#         y_true: True responses, a float numpy array of shape (num_samples, num_tasks). 
#         y_pred: Predicted responses, a float numpy array of shape (num_samples, num_tasks).
#         sample_weight: Sample weights, a float numpy array of shape (num_samples, ).
        
#     Returns:
#         accs: accuracy, a float numpy array of shape (num_tasks,).
#         balanced_accs: balanced accuracy, a float numpy array of shape (num_tasks,).
#     """
#     _, num_tasks = y_true.shape
#     thresholds = np.max(y_pred, axis=0)+1e-6
#     y_trues = np.array_split(y_true, num_tasks, axis=1)
#     y_preds = np.array_split(y_pred, num_tasks, axis=1)
#     accs = [0.0]*num_tasks
#     accs_thresholds = [0.0]*num_tasks
#     balanced_accs = [0.0]*num_tasks
#     balanced_accs_thresholds = [0.0]*num_tasks
#     y_ts = [yt.reshape(-1) for yt in y_trues]
#     y_ps = [yp.reshape(-1) for yp in y_preds]
#     sws = [sample_weight.reshape(-1)]*num_tasks
    
#     sws = [sw[~np.isnan(yt)] for sw, yt in zip(sws, y_ts)]
#     y_ps = [yp[~np.isnan(yt)] for yp, yt in zip(y_ps, y_ts)]
#     y_ts = [yt[~np.isnan(yt)] for yt in y_ts]

#     sws = [sw/np.sum(sw) for sw in sws]
#     percentiles = np.linspace(start=100, stop=80, num=100, endpoint=True)
#     for i in range(num_tasks):
#         acc = np.array([metrics.accuracy_score(y_ts[i], y_ps[i]>np.percentile(y_ps[i], per, axis=0), sample_weight=sws[i]) for per in percentiles])
#         balanced_acc = np.array([metrics.balanced_accuracy_score(y_ts[i], y_ps[i]>np.percentile(y_ps[i], per, axis=0), sample_weight=sws[i]) for per in percentiles])
        
#         acc_index = np.argmax(acc)
#         accs[i] = acc[acc_index]
#         thresholds = [np.percentile(y_ps[i], per, axis=0) for per in percentiles]
#         accs_thresholds[i] = thresholds[acc_index]

#         balanced_acc_index = np.argmax(balanced_acc)
#         balanced_accs[i] = balanced_acc[balanced_acc_index]
#         thresholds = [np.percentile(y_ps[i], per, axis=0) for per in percentiles]
#         balanced_accs_thresholds[i] = thresholds[balanced_acc_index]
                        
#     return accs, accs_thresholds, balanced_accs, balanced_accs_thresholds

def weighted_gini(act,pred,weight=None):
    if weight is None:
        weight    = np.ones(len(act))
    df            = pd.DataFrame({"act":act,"pred":pred,"weight":weight})
    df            = np.vstack((act, pred, weight))
#         sort_idx      = np.argsort(df[1])[::-1]
    sort_idx      = np.argsort(df[1]) # ascending to match definition in doc
    df            = df[:, sort_idx]
    random        = np.cumsum(df[2] / np.sum(df[2]))
    total_pos     = np.sum(df[0] * df[2])
    cum_pos_found = np.cumsum(df[0] * df[2])
    lorentz       = cum_pos_found / total_pos
#    lorentz      = np.where(total_pos>0, cum_pos_found / total_pos, 0)
    gini          = np.sum(lorentz[1:] * random[:-1]) - np.sum(lorentz[:-1] * (random[1:]))
    return gini

def normalized_weighted_gini(act,pred,weight=None):
    if weight is None:
        weight = len(act)*[1]
    return weighted_gini(act,pred,weight=weight) / weighted_gini(act,act,weight=weight) 


def multitask_masked_normalized_weighted_gini(y_true, y_pred, sample_weight):
    """Computes normalized weighted gini for multitask with missing responses.
    
    Args:
        y_true: True responses, a numpy array of shape (num_samples, num_tasks). 
        y_pred: Predicted responses, a numpy array of shape (num_samples, num_tasks).
        sample_weight: Sample weights, a numpy array of shape (num_samples, ).
    
    Returns:
        weighted normalized gini, float scalar, a float numpy array of shape (num_tasks,).
    """
    
    _, num_tasks = y_true.shape
    gini = []
    for yt, yp in zip(np.array_split(y_true, num_tasks, axis=1),
                      np.array_split(y_pred, num_tasks, axis=1)):
        yt = yt.reshape(-1)
        yp = yp.reshape(-1)
        sw = deepcopy(sample_weight)
        print(yt.shape, yp.shape, sw.shape)
        sw = sw[~np.isnan(yt)]
        yp = yp[~np.isnan(yt)]
        yt = yt[~np.isnan(yt)]        
        assert (yt>=0.0).all(), "True responses must be non-negative"
        assert (yp>=0.0).all(), "Predicted responses must be non-negative"
        gini.append(normalized_weighted_gini(yt, yp, weight=sw))
    return gini


def multitask_masked_normalized_weighted_gini_percent(y_true, y_pred, sample_weight, per=0.05):
    """Computes normalized weighted gini for multitask with missing responses.
    
    Args:
        y_true: True responses, a numpy array of shape (num_samples, num_tasks). 
        y_pred: Predicted responses, a numpy array of shape (num_samples, num_tasks).
        sample_weight: Sample weights, a numpy array of shape (num_samples, ).
    
    Returns:
        weighted normalized gini, float scalar, a float numpy array of shape (num_tasks,).
    """
    
    _, num_tasks = y_true.shape
    gini = []
    for yt, yp in zip(np.array_split(y_true, num_tasks, axis=1),
                      np.array_split(y_pred, num_tasks, axis=1)):
        yt = yt.reshape(-1)
        yp = yp.reshape(-1)
        sw = deepcopy(sample_weight)
        sw = sw[~np.isnan(yt)]
        yp = yp[~np.isnan(yt)]
        yt = yt[~np.isnan(yt)] 
        assert (yt>=0.0).all(), "True responses must be non-negative"
        assert (yp>=0.0).all(), "Predicted responses must be non-negative"
        sw_nz = sw[yt!=0] 
        yp_nz = yp[yt!=0] 
        yt_nz = yt[yt!=0] 
        sw_z = sw[yt==0] 
        yp_z = yp[yt==0]
        yt_z = yt[yt==0]
        indices = np.argsort(yt_nz)
        sw_nz = sw_nz[indices]
        yp_nz = yp_nz[indices]
        yt_nz = yt_nz[indices]
        num_responses_remove = (int)(len(indices)*per)
        sw_nz = sw_nz[:-num_responses_remove]
        yp_nz = yp_nz[:-num_responses_remove]
        yt_nz = yt_nz[:-num_responses_remove]
        yt_c = np.hstack([yt_z, yt_nz])
        yp_c = np.hstack([yp_z, yp_nz])
        sw_c = np.hstack([sw_z, sw_nz])
        gini.append(normalized_weighted_gini(yt_c, yp_c, weight=sw_c))
    return gini

def multitask_masked_normalized_weighted_gini_nonzeros(y_true, y_pred, sample_weight):
    """Computes normalized weighted gini for multitask with missing responses.
    
    Args:
        y_true: True responses, a numpy array of shape (num_samples, num_tasks). 
        y_pred: Predicted responses, a numpy array of shape (num_samples, num_tasks).
        sample_weight: Sample weights, a numpy array of shape (num_samples, ).
    
    Returns:
        weighted normalized gini, float scalar, a float numpy array of shape (num_tasks,).
    """
    
    _, num_tasks = y_true.shape
    gini_nonzeros = []
    for yt, yp in zip(np.array_split(y_true, num_tasks, axis=1),
                      np.array_split(y_pred, num_tasks, axis=1)):
        yt = yt.reshape(-1)
        yp = yp.reshape(-1)
        sw = deepcopy(sample_weight)
        sw = sw[~np.isnan(yt)]
        yp = yp[~np.isnan(yt)]
        yt = yt[~np.isnan(yt)] 
        assert (yt>=0.0).all(), "True responses must be non-negative"
        assert (yp>=0.0).all(), "Predicted responses must be non-negative"
        sw_nz = sw[yt!=0] 
        yp_nz = yp[yt!=0] 
        yt_nz = yt[yt!=0] 
        gini_nonzeros.append(normalized_weighted_gini(yt_nz, yp_nz, weight=sw_nz))
    return gini_nonzeros


def compute_task_relatedness(response, task_nums, visualize=True):
    """Computes task relations between responses using pearson correlation and mutual information.
    
    Args:
        responses: target responses.
        task_nums: task numbers for labelling, list of ints.
    
    Returns:
        corr_mx: pairwise pearson correlation between responses.
        mi_mx: pairwise mutual information between responses.
    """
    corr_mx = np.zeros((len(task_nums), len(task_nums)))
    mi_mx = np.zeros((len(task_nums), len(task_nums)))
    
    for i in range(len(task_nums)):
        for j in range(len(task_nums)):
            if i==j:
                continue
            y_i = response[:,i]
            y_j = response[:,j]
            y = np.vstack([y_i, y_j]).T
            y_i = y_i[(~np.isnan(y)).sum(axis=1)==2]
            y_j = y_j[(~np.isnan(y)).sum(axis=1)==2]
            corr_mx[i,j] = np.absolute(pearsonr(y_i, y_j)[0])
            mi_mx[i,j] = mutual_info_regression(y_i.reshape(-1, 1), y_j)
    
    mi_mx = 0.5*(mi_mx+mi_mx.T) # symmetrize as mutual information is supposed to be symmetric! 
    
    if visualize:
        font = {'weight' : 'normal',
                'size'   : 14}
        plt.rc('font', **font)
        column_names = ['T'+str(i+1) for i in task_nums]

        plt.figure(figsize=(10,10))
        sns.heatmap(mi_mx)
        plt.xticks(task_nums, column_names, rotation=90)
        plt.yticks(task_nums, column_names, rotation=0)
        plt.show()

        plt.figure(figsize=(10,10))
        sns.heatmap(corr_mx)
        plt.xticks(task_nums, column_names, rotation=90)
        plt.yticks(task_nums, column_names, rotation=0)
        plt.show() 
    
    return corr_mx, mi_mx

# Hussein's modified version of the poisson_deviance function which handles preds = 0 and actual != 0. Assumes no NaN responses.
import warnings
# A quick hack to supress division by zero in poisson_deviance. This is OK
# since the case of actual==0 is handled separately in poisson_deviance.
warnings.filterwarnings("ignore")

def singletask_poisson_deviance(actual, preds, wgt):
    poisson_dev = np.sum(
        np.sum(
            np.where(actual==0,
                     (-actual + preds)*wgt,
                     (actual*np.log((actual)/(preds+1e-6)) - actual + preds)*wgt ), axis=0) / np.sum(wgt, axis=0)
    )
    return poisson_dev

def singletask_mean_squared_error(true, pred, weight):
    diff = ((true - pred)**2) * weight
    return np.sum(diff / np.sum(weight, axis=0, keepdims=True))