import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K

import losses
import models_multitask
import utils_multitask
import layers
import soft_trees
from tensorflow.python.client import device_lib

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

_DUMMY_RESPONSE = 1e8

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def get_available_cpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'CPU']

def objective(
    trial,
    data_processed,
    loss_dist,
    tuning_criteria,
    max_epochs,
    path,
    architecture='shared',
    activation='sigmoid',
    max_trees=100,
    max_depth=4,
    ):
    
    """ Clear the backend (TensorFlow). See:
    https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session
    """
    K.clear_session() 
    
    num_tasks = data_processed.y_train_processed.shape[1]
    print("====================num_tasks", num_tasks)
    print("====================path", path)
    
    ### Soft Decision Tree parameters 
    num_trees = trial.suggest_int('num_trees', 1, max_trees)
    depth = trial.suggest_int('depth', 2, max_depth)
    # We only use a single layer of Tree Ensemble [Note layer is different than depth]. 
    num_layers = trial.suggest_int('num_layers', 1, 1)
    # kernel_l2 = trial.suggest_loguniform('kernel_l2', 1e-5, 1e-0)
    kernel_l2 = trial.suggest_categorical('kernel_l2', [0.0])
    
    activation = trial.suggest_categorical('activation', [activation])
    if activation=='sigmoid':
        activation==tf.keras.activations.sigmoid
    elif activation=='smooth-step':
        gamma = trial.suggest_loguniform('gamma', 1e-4, 10)
        activation==tf.keras.layers.Activation(soft_trees.SmoothStep(gamma), name='SmoothStep')
        
    ### Loss parameters
    task_weights = np.ones(num_tasks)
    loss_criteria = trial.suggest_categorical('loss_criteria', [loss_dist])
    # exponentials and sigmoids inside distributions
    if loss_criteria in ['mse', 'poisson', 'zero-inflated-poisson', 'negative-binomial', 'zero-inflated-negative-binomial']:
        output_activation = trial.suggest_categorical('output_activation', ['linear'])
    loss = losses.NegativeLogLikelihood()
    
    # Whether to share regression and classification Tree Ensemble between classification and regression
    if loss_criteria in ['zero-inflated-poisson', 'zero-inflated-negative-binomial']: 
        architecture = trial.suggest_categorical('architecture', [architecture])
    
    ### Optimization parameters
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512])
    optimizer = trial.suggest_categorical('optimizer', ['adam'])
    if optimizer=='adam':
        if loss_criteria in ['mse', 'poisson', 'binary']:
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
            optim = tf.keras.optimizers.Adam(learning_rate)
        elif loss_criteria in ['zero-inflated-poisson', 'zero-inflated-negative-binomial']: 
            learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
            optim = tf.keras.optimizers.Adam(learning_rate, epsilon=5e-5) 
    elif optimizer=='rmsprop':
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1e-2)
        optim = tf.keras.optimizers.RMSprop(learning_rate)
    epochs = max_epochs
    # epochs = trial.suggest_int('epochs', 20, max_epochs, 5)
    model_type = trial.suggest_categorical('model_type', [None])

    y_train_processed = deepcopy(data_processed.y_train_processed)
    sample_weights = data_processed.w_train.reshape(-1,1)[~np.isnan(y_train_processed)[:,0],:] 
    x_train_processed = data_processed.x_train_processed[~np.isnan(y_train_processed)[:,0],:]
    y_train_processed = y_train_processed[~np.isnan(y_train_processed)[:,0],:]
    print(x_train_processed.shape, y_train_processed.shape, sample_weights.shape)
    
#     # Create a MirroredStrategy.
#     strategy = tf.distribute.MirroredStrategy(
#         devices=["/gpu:0","/gpu:2","/gpu:3","/gpu:4"],
#     )
#     print("Number of devices: {}".format(strategy.num_replicas_in_sync))

#     # Open a strategy scope.
#     with strategy.scope():
#         # Everything that creates variables should be under the strategy scope.
#         # In general this is only model construction & `compile()`.


    if loss_criteria in ['mse', 'poisson']:
        leaf_dims = (num_tasks, )
        x = tf.keras.layers.Input(name='input', shape=data_processed.x_train_processed.shape[1:])
        submodel = models_multitask.create_multitask_submodel(
            x,
            num_layers,
            num_trees,
            depth,
            num_tasks,
            leaf_dims,
            "MultitaskRegression",
            activation=activation,
            model_type=model_type,
            kernel_regularizer=tf.keras.regularizers.L2(kernel_l2)
        )
        x = submodel.input
        outputs = submodel(x)
        outputs = tf.split(outputs, num_or_size_splits=num_tasks, axis=1)
        ypreds = []
        sample_weights_dict = {}
        for i, rp in enumerate(outputs):
            ypred = tf.keras.layers.Activation(output_activation)(rp)
            if loss_criteria=='mse':
                ypred = tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.Normal(t[..., 0], scale=1.0),
                    name='task{}'.format(i)
                )(ypred)
#                 ypred = tfp.layers.DistributionLambda(tfp.distributions.Normal, name='task{}'.format(i))(tf.squeeze(ypred))
            elif loss_criteria=='poisson':
                ypred = tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.Poisson(log_rate=t[..., 0]),
                    name='task{}'.format(i)
                )(ypred)
#                 ypred = tfp.layers.DistributionLambda(tfp.distributions.Poisson, name='task{}'.format(i))(tf.squeeze(ypred))
            ypreds.append(ypred)
            sample_weights_dict['task{}'.format(i)]=sample_weights[:,i]*task_weights[i]            
        model = tf.keras.Model(inputs=x, outputs=ypreds)
        model.summary()
    elif loss_criteria=='zero-inflated-poisson':
        if architecture=='separate':
            leaf_dims = (num_tasks, )
            x = tf.keras.layers.Input(name='input', shape=data_processed.x_train_processed.shape[1:])
            submodel = models_multitask.create_multitask_submodel(
                x,
                num_layers,
                num_trees,
                depth,
                num_tasks,
                leaf_dims,
                "MultitaskRegression",
                activation=activation,
                model_type=model_type,
                kernel_regularizer=tf.keras.regularizers.L2(kernel_l2)
            )
            x = submodel.input
            outputs = submodel(x)
            regression_preds = tf.keras.layers.Activation(output_activation)(outputs)
            sub_classification_model = models_multitask.create_multitask_submodel(
                x,
                num_layers,
                num_trees,
                depth,
                num_tasks,
                leaf_dims,
                "MultitaskClassification",
                activation=activation,
                model_type=model_type,
                kernel_regularizer=tf.keras.regularizers.L2(kernel_l2)                
            )
            classification_logodds = sub_classification_model(x)
            regression_preds = tf.split(regression_preds, num_or_size_splits=num_tasks, axis=1)
            classification_logodds = tf.split(classification_logodds, num_or_size_splits=num_tasks, axis=1)

        elif architecture=='shared':
            leaf_dims = (2*num_tasks, )
            x = tf.keras.layers.Input(name='input', shape=data_processed.x_train_processed.shape[1:])
            submodel = models_multitask.create_multitask_submodel(
                x,
                num_layers,
                num_trees,
                depth,
                2*num_tasks,
                leaf_dims,
                "MultitaskRegressionClassification",
                activation=activation,
                model_type=model_type,
                kernel_regularizer=tf.keras.regularizers.L2(kernel_l2)
            )
            x = submodel.input
            outputs = submodel(x)
            regression_preds,  classification_logodds= tf.split(outputs, num_or_size_splits=2, axis=1)
            regression_preds = tf.keras.layers.Activation(output_activation, name="MultitaskRegression")(regression_preds)
            classification_logodds = tf.keras.layers.Activation('linear', name="MultitaskClassification")(classification_logodds)
            regression_preds = tf.split(regression_preds, num_or_size_splits=num_tasks, axis=1)
            classification_logodds = tf.split(classification_logodds, num_or_size_splits=num_tasks, axis=1)
            
        ypreds = []
        sample_weights_dict = {}
        for i, (rp, cp) in enumerate(zip(regression_preds, classification_logodds)):
            ypred = tf.keras.layers.Concatenate(axis=1)([rp, cp])
            ypred = tfp.layers.DistributionLambda(layers.zero_inflation_mixture_singletask, name='task{}'.format(i))(ypred)
            ypreds.append(ypred)

            sample_weights_dict['task{}'.format(i)]=sample_weights[:,i]*task_weights[i]
        model = tf.keras.Model(inputs=x, outputs=ypreds)
    elif loss_criteria in ['negative-binomial']:
        leaf_dims = (2*num_tasks, )
        x = tf.keras.layers.Input(name='input', shape=data_processed.x_train_processed.shape[1:])
        submodel = models_multitask.create_multitask_submodel(
            x,
            num_layers,
            num_trees,
            depth,
            2*num_tasks,
            leaf_dims,
            "MultitaskRegression",
            activation=activation,
            model_type=model_type,
            kernel_regularizer=tf.keras.regularizers.L2(kernel_l2)
        )
        x = submodel.input
        outputs = submodel(x)
        outputs = tf.split(outputs, num_or_size_splits=num_tasks, axis=1)
        ypreds = []
        sample_weights_dict = {}
        for i, rp in enumerate(outputs):
            ypred = tf.keras.layers.Activation(output_activation)(rp)
            ypred = tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.NegativeBinomial.experimental_from_mean_dispersion(
                    mean=tf.math.exp(t[..., 0]), dispersion=tf.math.sigmoid(t[..., 1])
                ),
                name='task{}'.format(i)
            )(ypred)
            ypreds.append(ypred)
            sample_weights_dict['task{}'.format(i)]=sample_weights[:,i]*task_weights[i]            
        model = tf.keras.Model(inputs=x, outputs=ypreds)
    elif loss_criteria=='zero-inflated-negative-binomial':
        if architecture=='separate':
            leaf_dims = (2*num_tasks, )
            x = tf.keras.layers.Input(name='input', shape=data_processed.x_train_processed.shape[1:])
            submodel = models_multitask.create_multitask_submodel(
                x,
                num_layers,
                num_trees,
                depth,
                2*num_tasks,
                leaf_dims,
                "MultitaskRegression",
                activation=activation,
                model_type=model_type,
                kernel_regularizer=tf.keras.regularizers.L2(kernel_l2)
            )
            x = submodel.input
            outputs = submodel(x)
            regression_preds = tf.keras.layers.Activation(output_activation)(outputs)
            
            leaf_dims = (num_tasks, )
            sub_classification_model = models_multitask.create_multitask_submodel(
                x,
                num_layers,
                num_trees,
                depth,
                2*num_tasks,
                leaf_dims,
                "MultitaskClassification",
                activation=activation,
                model_type=model_type,
                kernel_regularizer=tf.keras.regularizers.L2(kernel_l2)
            )
            classification_logodds = sub_classification_model(x)
            regression_preds = tf.split(regression_preds, num_or_size_splits=num_tasks, axis=1)
            classification_logodds = tf.split(classification_logodds, num_or_size_splits=num_tasks, axis=1)

        elif architecture=='shared':
            leaf_dims = (3*num_tasks, )
            x = tf.keras.layers.Input(name='input', shape=data_processed.x_train_processed.shape[1:])
            submodel = models_multitask.create_multitask_submodel(
                x,
                num_layers,
                num_trees,
                depth,
                3*num_tasks,
                leaf_dims,
                "MultitaskRegressionClassification",
                activation=activation,
                model_type=model_type,
                kernel_regularizer=tf.keras.regularizers.L2(kernel_l2)
            )
            x = submodel.input
            outputs = submodel(x)
            regression_preds,  classification_logodds= tf.split(outputs, num_or_size_splits=[2*num_tasks, num_tasks], axis=1)
            regression_preds = tf.keras.layers.Activation(output_activation, name="MultitaskRegression")(regression_preds)
            classification_logodds = tf.keras.layers.Activation('linear', name="MultitaskClassification")(classification_logodds)
            regression_preds = tf.split(regression_preds, num_or_size_splits=num_tasks, axis=1)
            classification_logodds = tf.split(classification_logodds, num_or_size_splits=num_tasks, axis=1)
            
        ypreds = []
        sample_weights_dict = {}
        for i, (rp, cp) in enumerate(zip(regression_preds, classification_logodds)):
            ypred = tf.keras.layers.Concatenate(axis=1)([rp, cp])
            ypred = tfp.layers.DistributionLambda(
                layers.zero_inflation_negative_binomial_mixture_singletask,
                name='task{}'.format(i)
            )(ypred)
            ypreds.append(ypred)

            sample_weights_dict['task{}'.format(i)]=sample_weights[:,i]*task_weights[i]
        model = tf.keras.Model(inputs=x, outputs=ypreds)
    elif loss_criteria=='binary':
        leaf_dims = (num_tasks, )
        x = tf.keras.layers.Input(name='input', shape=data_processed.x_train_processed.shape[1:])
        submodel = models_multitask.create_multitask_submodel(
            x,
            num_layers,
            num_trees,
            depth,
            num_tasks,
            leaf_dims,
            "Classification",
            activation=activation,
            model_type=model_type,
            kernel_regularizer=tf.keras.regularizers.L2(kernel_l2)
        )
        x = submodel.input
        outputs = submodel(x)
        print(outputs)
        outputs = tf.split(outputs, num_or_size_splits=num_tasks, axis=1)
        print(outputs)
        ypreds = []
        sample_weights_dict = {}
        for i, rp in enumerate(outputs):
            ypred = tf.keras.layers.Activation('sigmoid', name='task{}'.format(i))(rp)
            ypreds.append(ypred)
            print(ypreds)
            sample_weights_dict['task{}'.format(i)]=sample_weights[:,i]*task_weights[i]            
            loss = 'binary_crossentropy'
        model = tf.keras.Model(inputs=x, outputs=ypreds)
        model.summary()
        
#     model.summary()

    model.compile(loss=loss, optimizer=optim)

        
    y_valid_processed = deepcopy(data_processed.y_valid_processed)
    sample_weights_valid = data_processed.w_valid.reshape(-1,1)[~np.isnan(y_valid_processed)[:,0],:] 
    x_valid_processed = data_processed.x_valid_processed[~np.isnan(y_valid_processed)[:,0],:]
    y_valid_processed = y_valid_processed[~np.isnan(y_valid_processed)[:,0],:]
    y_valid_processed = y_valid_processed.astype('float32')
    sample_weights_valid = sample_weights_valid.astype('float32')
    sample_weights_valid_dict = {}
    for i in range(num_tasks):
        sample_weights_valid_dict['task{}'.format(i)]=sample_weights_valid[:,i]*task_weights[i]            
    # print(y_valid_pred_dist)

    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=25, verbose=1, mode='auto', restore_best_weights=True)
    ]
    with tf.device(get_available_gpus()[0]):
        history = model.fit(x=x_train_processed, 
                  y=tf.split(y_train_processed, num_or_size_splits=num_tasks, axis=1),
                  sample_weight=sample_weights_dict,
                  epochs=epochs, 
                  batch_size=batch_size, 
                  shuffle=True,
                  callbacks=callbacks,
                  validation_data=(x_valid_processed, tf.split(y_valid_processed, num_or_size_splits=num_tasks, axis=1), sample_weights_valid_dict),
                  verbose=0, 
                  )  
        number_of_epochs_it_ran = len(history.history['loss'])

    with tf.device(get_available_cpus()[0]):
        # Check for infinite loss
        training_loss = model.evaluate(x_train_processed,
                                       tf.split(y_train_processed, num_or_size_splits=num_tasks, axis=1),
                                       sample_weight=sample_weights_dict,
                                       batch_size=batch_size,
                                       verbose=0)

        if np.isfinite(np.sum(training_loss)) or ~np.isnan(np.sum(training_loss)):
            # Evaluation
            y_valid_processed = deepcopy(data_processed.y_valid_processed)
            sample_weights_valid = ~np.isnan(y_valid_processed)*data_processed.w_valid.reshape(-1,1)
            y_valid_processed[np.isnan(y_valid_processed)] = _DUMMY_RESPONSE
            y_valid_processed = y_valid_processed.astype('float32')
            sample_weights_valid = sample_weights_valid.astype('float32')
            
            y_valid_pred_dist = model(data_processed.x_valid_processed) 
            y_test_pred_dist = model(data_processed.x_test_processed)
            if num_tasks==1:
                y_valid_pred_dist = [y_valid_pred_dist]
                y_test_pred_dist = [y_test_pred_dist]

            ########################### Compute evaluation metrics #########################            
            if loss_criteria in ['mse', 'poisson', 'zero-inflated-poisson']:
                y_valid_pred = tf.stack([yv.mean() for yv in y_valid_pred_dist], axis=1).numpy() # compute expectation
                y_test_pred = tf.stack([yv.mean() for yv in y_test_pred_dist], axis=1).numpy() # compute expectation
            elif loss_criteria in ['binary']:
                y_valid_pred = tf.stack([tf.reshape(yv, [-1]) for yv in y_valid_pred_dist], axis=1).numpy() # compute expectation
                y_test_pred = tf.stack([tf.reshape(yv, [-1]) for yv in y_test_pred_dist], axis=1).numpy() # compute expectation
            
            # print(y_valid_pred.shape)
            # print(y_valid_pred)
            # Classification metrics
            # AUC based on expectations
            y_valid_classes = utils_multitask.multitask_masked_convert_to_binary_labels(data_processed.y_valid_processed)
            roc_auc_exps_valid = utils_multitask.multitask_masked_roc_auc(y_valid_classes, y_valid_pred, data_processed.w_valid)
            print("roc_auc_exps (valid):", roc_auc_exps_valid)   
            # print(y_valid_classes.shape)
            # print(y_valid_classes)
            acc_exps_valid, acc_thresholds_exps_valid, balanced_acc_exps_valid, balanced_acc_thresholds_exps_valid = utils_multitask.multitask_masked_accuracy(y_valid_classes, y_valid_pred, data_processed.w_valid)
            print("acc_exps (valid):", acc_exps_valid)
            print("balanced_acc_exps (valid):", balanced_acc_exps_valid)
            
            num_tests = len(data_processed.y_tests_processed)
            # roc_auc_exps_tests = [None]*num_tests
            # acc_exps_tests = [None]*num_tests
            # balanced_acc_exps_tests = [None]*num_tests
            # for j, y_test_processed in enumerate(data_processed.y_tests_processed):
            #     y_test_classes = utils_multitask.multitask_masked_convert_to_binary_labels(y_test_processed)
            #     roc_auc_exps_tests[j] = utils_multitask.multitask_masked_roc_auc(y_test_classes, y_test_pred, data_processed.w_test)
            #     acc_exps_tests[j], _, balanced_acc_exps_tests[j], _ = utils_multitask.multitask_masked_accuracy(y_test_classes, y_test_pred, data_processed.w_test, acc_thresholds=acc_thresholds_exps_valid, bal_acc_thresholds=balanced_acc_thresholds_exps_valid)

            # # AUC based on probabilities
            # zeros_probs_valid = np.hstack([yp.prob(np.zeros((yp.shape[0],))).numpy().reshape(-1,1) for yp in y_valid_pred_dist])
            # zeros_probs_test = np.hstack([yp.prob(np.zeros((yp.shape[0],))).numpy().reshape(-1,1) for yp in y_test_pred_dist])
            # nonzeros_probs_valid = 1-zeros_probs_valid
            # nonzeros_probs_test = 1-zeros_probs_test
            # roc_auc_probs_valid = utils_multitask.multitask_masked_roc_auc(y_valid_classes, nonzeros_probs_valid, data_processed.w_valid)
            # roc_auc_probs_test = utils_multitask.multitask_masked_roc_auc(y_test_classes, nonzeros_probs_test, data_processed.w_test)
            # print("roc_auc_probs (valid):", roc_auc_probs_valid)        
            # acc_probs_valid, acc_thresholds_probs_valid, balanced_acc_probs_valid, balanced_acc_thresholds_probs_valid = utils_multitask.multitask_masked_accuracy(y_valid_classes, nonzeros_probs_valid, data_processed.w_valid)
            # acc_probs_test, _, balanced_acc_probs_test, _ = utils_multitask.multitask_masked_accuracy(y_test_classes, nonzeros_probs_test, data_processed.w_test)
            # print("acc_probs (valid):", acc_probs_valid)
            # print("balanced_acc_probs (valid):", balanced_acc_probs_valid)
            
            if loss_criteria in ['mse', 'poisson', 'zero-inflated-poisson']:
                negloglikelihood_valid = [losses.NegativeLogLikelihood()(yv, yvp, sw).numpy() for yv, yvp, sw in zip(np.array_split(y_valid_processed, num_tasks, axis=1), y_valid_pred_dist, np.array_split(sample_weights_valid, num_tasks, axis=1))]
                print("negloglikelihood (valid):", negloglikelihood_valid)

                mse_valid = utils_multitask.multitask_masked_mean_squared_error(data_processed.y_valid_processed, y_valid_pred, data_processed.w_valid)
                print('mse (valid):', mse_valid)

                poisson_deviance_valid, poisson_deviance_zeros_valid, poisson_deviance_nonzeros_valid = utils_multitask.multitask_masked_poisson_deviance(data_processed.y_valid_processed, y_valid_pred, data_processed.w_valid)
                print('poisson_deviance (valid):', poisson_deviance_valid)
                
                num_tests = len(data_processed.y_tests_processed)
                negloglikelihood_tests = [None]*num_tests
                mse_tests = [None]*num_tests
                poisson_deviance_tests = [None]*num_tests
                poisson_deviance_zeros_tests = [None]*num_tests
                poisson_deviance_nonzeros_tests = [None]*num_tests
                for j, y_test_processed in enumerate(data_processed.y_tests_processed):
                    y_test_processed = deepcopy(y_test_processed)
                    sample_weights_test = ~np.isnan(y_test_processed)*data_processed.w_test.reshape(-1,1)
                    y_test_processed[np.isnan(y_test_processed)] = _DUMMY_RESPONSE
                    y_test_processed = y_test_processed.astype('float32')
                    sample_weights_test = sample_weights_test.astype('float32')        

                    negloglikelihood_tests[j] = [losses.NegativeLogLikelihood()(yv, yvp, sw).numpy() for yv, yvp, sw in zip(np.array_split(y_test_processed, num_tasks, axis=1), y_test_pred_dist, np.array_split(sample_weights_test, num_tasks, axis=1))]
                for j, y_test_processed in enumerate(data_processed.y_tests_processed):
                    y_test_processed = deepcopy(y_test_processed)
                    mse_tests[j] = utils_multitask.multitask_masked_mean_squared_error(y_test_processed, y_test_pred, data_processed.w_test)
                    

    #             gini_valid = utils_multitask.multitask_masked_normalized_weighted_gini(data_processed.y_valid_processed, y_valid_pred, data_processed.w_valid)
    #             gini_test = utils_multitask.multitask_masked_normalized_weighted_gini(data_processed.y_test_processed, y_test_pred, data_processed.w_test)
    #             print("gini (valid):", gini_valid)

                    poisson_deviance_tests[j], poisson_deviance_zeros_tests[j], poisson_deviance_nonzeros_tests[j] = utils_multitask.multitask_masked_poisson_deviance(y_test_processed, y_test_pred, data_processed.w_test)
    #             poisson_deviance_thresholds_valid, scalers_valid, percentiles_valid, y_valid_pred_scaled = utils_multitask.multitask_masked_poisson_deviance_thresholds(data_processed.y_valid_processed, y_valid_pred, data_processed.w_valid)
    #             scaled_poisson_deviance_valid, scaled_poisson_deviance_zeros_valid, scaled_poisson_deviance_nonzeros_valid = utils_multitask.multitask_masked_poisson_deviance(data_processed.y_valid_processed, y_valid_pred_scaled, data_processed.w_valid)
    #             print('scaled_poisson_deviance (valid):', scaled_poisson_deviance_valid)
    #             print("scalers (valid):", scalers_valid)
    #             print("poisson_deviance_thresholds (valid):", poisson_deviance_thresholds_valid)
    #             print("percentiles (valid):", percentiles_valid)
    #             y_test_pred_scaled = deepcopy(y_test_pred)
    #             for i in range(num_tasks):
    #                 y_test_pred_scaled[y_test_pred[:,i]>poisson_deviance_thresholds_valid[i],:] *= scalers_valid[i]
    #             scaled_poisson_deviance_test, scaled_poisson_deviance_zeros_test, scaled_poisson_deviance_nonzeros_test = utils_multitask.multitask_masked_poisson_deviance(data_processed.y_test_processed, y_test_pred_scaled, data_processed.w_test)

        else:
            if loss_criteria in ['mse', 'poisson', 'zero-inflated-poisson']: 
                num_tests = len(data_processed.y_tests_processed)
                negloglikelihood_valid = [np.inf]*num_tasks
                negloglikelihood_tests = [[np.inf]*num_tests]*num_tasks
                mse_valid = [np.inf]*num_tasks
                mse_tests = [[np.inf]*num_tests]*num_tasks
                poisson_deviance_valid, poisson_deviance_zeros_valid, poisson_deviance_nonzeros_valid = ([np.inf]*num_tasks, [np.inf]*num_tasks, [np.inf]*num_tasks)
                poisson_deviance_tests, poisson_deviance_zeros_tests, poisson_deviance_nonzeros_tests = ([[np.inf]*num_tests]*num_tasks, [[np.inf]*num_tests]*num_tasks, [[np.inf]*num_tests]*num_tasks)
                # scalers_valid = [0.0]*num_tasks
                # poisson_deviance_thresholds_valid = [np.inf]*num_tasks
                # percentiles_valid = [105]*num_tasks
                # scaled_poisson_deviance_valid, scaled_poisson_deviance_zeros_valid, scaled_poisson_deviance_nonzeros_valid = ([np.inf]*num_tasks, [np.inf]*num_tasks, [np.inf]*num_tasks)
                # scaled_poisson_deviance_test, scaled_poisson_deviance_zeros_test, scaled_poisson_deviance_nonzeros_test = ([np.inf]*num_tasks, [np.inf]*num_tasks, [np.inf]*num_tasks)
                # gini_valid = [0.0]*num_tasks
                # gini_test = [0.0]*num_tasks
            # roc_auc_exps_valid = [0.0]*num_tasks
            # roc_auc_exps_tests = [[0.0]*num_tests]*num_tasks
            # acc_exps_valid = [0.0]*num_tasks
            # acc_exps_tests = [[0.0]*num_tests]*num_tasks
            # balanced_acc_exps_valid = [0.0]*num_tasks
            # balanced_acc_exps_tests = [[0.0]*num_tests]*num_tasks
            # acc_thresholds_exps_valid = [np.inf]*num_tasks
            # balanced_acc_thresholds_exps_valid = [np.inf]*num_tasks
            # roc_auc_probs_valid = [0.0]*num_tasks
            # roc_auc_probs_test = [0.0]*num_tasks
            # acc_probs_valid = [0.0]*num_tasks
            # acc_probs_test = [0.0]*num_tasks
            # balanced_acc_probs_valid = [0.0]*num_tasks
            # balanced_acc_probs_test = [0.0]*num_tasks
            # acc_thresholds_probs_valid = [np.inf]*num_tasks
            # balanced_acc_thresholds_probs_valid = [np.inf]*num_tasks
    
    if loss_criteria in ['mse', 'poisson', 'zero-inflated-poisson']:   
        valid_criteria_multitask = np.sum(negloglikelihood_valid)
    elif loss_criteria in ['binary']:
        valid_criteria_multitask = np.sum(roc_auc_exps_valid)
    else:
        raise ValueError("loss criteria {} is not supported".format(loss_criteria))
        
    print('Valid Criteria:', valid_criteria_multitask)
    
    # Save trained model to a file.
    trial_path = os.path.join(path, "trials", "trial{}".format(trial.number))
    # model.save(os.path.join(trial_path, "model"))

    if loss_criteria in ['mse', 'poisson', 'zero-inflated-poisson']:
        trial.set_user_attr("negloglikelihood_valid", negloglikelihood_valid)
        trial.set_user_attr("negloglikelihood_tests", negloglikelihood_tests)
        trial.set_user_attr("mean_squared_error_valid", mse_valid)
        trial.set_user_attr("mean_squared_error_tests", mse_tests)
        trial.set_user_attr("poisson_deviance_valid", poisson_deviance_valid)
        trial.set_user_attr("poisson_deviance_tests", poisson_deviance_tests)
        trial.set_user_attr("poisson_deviance_zeros_valid", poisson_deviance_zeros_valid)
        trial.set_user_attr("poisson_deviance_zeros_tests", poisson_deviance_zeros_tests)
        trial.set_user_attr("poisson_deviance_nonzeros_valid", poisson_deviance_nonzeros_valid)
        trial.set_user_attr("poisson_deviance_nonzeros_tests", poisson_deviance_nonzeros_tests)
    # trial.set_user_attr("scalers_valid", scalers_valid)
    # trial.set_user_attr("poisson_deviance_thresholds_valid", poisson_deviance_thresholds_valid)
    # trial.set_user_attr("percentiles_valid", percentiles_valid)
    # trial.set_user_attr("scaled_poisson_deviance_valid", scaled_poisson_deviance_valid)
    # trial.set_user_attr("scaled_poisson_deviance_test", scaled_poisson_deviance_test)
    # trial.set_user_attr("scaled_poisson_deviance_zeros_valid", scaled_poisson_deviance_zeros_valid)
    # trial.set_user_attr("scaled_poisson_deviance_zeros_test", scaled_poisson_deviance_zeros_test)
    # trial.set_user_attr("scaled_poisson_deviance_nonzeros_valid", scaled_poisson_deviance_nonzeros_valid)
    # trial.set_user_attr("scaled_poisson_deviance_nonzeros_test", scaled_poisson_deviance_nonzeros_test)
    # trial.set_user_attr("gini_valid", gini_valid)
    # trial.set_user_attr("gini_test", gini_test)
    # trial.set_user_attr("roc_auc_exps_valid", roc_auc_exps_valid)
    # trial.set_user_attr("roc_auc_exps_tests", roc_auc_exps_tests)
    # trial.set_user_attr("acc_exps_valid", acc_exps_valid)
    # trial.set_user_attr("acc_exps_tests", acc_exps_tests)
    # trial.set_user_attr("balanced_acc_exps_valid", balanced_acc_exps_valid)
    # trial.set_user_attr("balanced_acc_exps_tests", balanced_acc_exps_tests)
    # trial.set_user_attr("acc_thresholds_exps_valid", acc_thresholds_exps_valid)
    # trial.set_user_attr("balanced_acc_thresholds_exps_valid", balanced_acc_thresholds_exps_valid)
    # trial.set_user_attr("roc_auc_probs_valid", roc_auc_probs_valid)
    # trial.set_user_attr("roc_auc_probs_test", roc_auc_probs_test)
    # trial.set_user_attr("acc_probs_valid", acc_probs_valid)
    # trial.set_user_attr("acc_probs_test", acc_probs_test)
    # trial.set_user_attr("balanced_acc_probs_valid", balanced_acc_probs_valid)
    # trial.set_user_attr("balanced_acc_probs_test", balanced_acc_probs_test)
    # trial.set_user_attr("acc_thresholds_probs_valid", acc_thresholds_probs_valid)
    # trial.set_user_attr("balanced_acc_thresholds_probs_valid", balanced_acc_thresholds_probs_valid)
    trial.set_user_attr("num_epochs", number_of_epochs_it_ran)
    
    return valid_criteria_multitask
