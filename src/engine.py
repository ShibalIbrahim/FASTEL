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
import pandas as pd
import numpy as np
import seaborn as sns
import data_utils
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

class MultiTaskTrees():
    def __init__(
        self,
        input_shape,
        loss_criteria='poisson',
        architecture='shared', # for ZIP and NB
        activation='sigmoid',
        output_activation='linear',
        num_trees=20,
        depth=3,
        num_tasks=2,
        model_type='regularized', # soft parameter sharing in multitask
        alpha=0.1, # strength of regularization for sharing in multitask,
        power=1.0, # 
        batch_size=64,
        learning_rate=0.001,
        epochs=200,
        dummy_response=_DUMMY_RESPONSE,
    ):
        self.input_shape = input_shape
        self.loss_criteria = loss_criteria
        self.activation = activation
        self.output_activation = output_activation
        self.num_trees = num_trees
        self.depth = depth
        self.num_tasks = num_tasks
        if self.num_tasks==1: # Single-Task Learning
            self.model_type = None
            self.architecture = 'shared'
        else: # Multi-Task Learning
            self.model_type = model_type
            self.architecture = 'shared-per-task'
        self.alpha = alpha
        self.power = power
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dummy_response = dummy_response
        self.loss = losses.NegativeLogLikelihood()
        self.task_weights = np.ones(self.num_tasks)
        
        ### Optimization parameters
        if loss_criteria in ['mse', 'poisson', 'negative-binomial']:
            optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        elif loss_criteria in ['zero-inflated-poisson']: 
            optimizer = tf.keras.optimizers.Adam(self.learning_rate, epsilon=5e-5)
        self.optimizer = optimizer
        self.model = self.create_model()
        
    def create_model(self):
        x = tf.keras.layers.Input(name='input', shape=self.input_shape)
        if self.loss_criteria in ['mse', 'poisson']:
            leaf_dims = (self.num_tasks, )
            submodel = models_multitask.create_multitask_submodel(
                x,
                self.num_trees,
                self.depth,
                self.num_tasks,
                leaf_dims,
                "MultitaskRegression",
                model_type=self.model_type,
                alpha=self.alpha,
                power=self.power,
            )
            x = submodel.input
            outputs = submodel(x)
            outputs = tf.split(outputs, num_or_size_splits=self.num_tasks, axis=1)
            ypreds = []
            for i, rp in enumerate(outputs):
                ypred = tf.keras.layers.Activation('linear')(rp)
                if loss_criteria=='mse':
                    ypred = tfp.layers.DistributionLambda(
                        lambda t: tfp.distributions.Normal(t[..., 0], scale=1.0),
                        name='task{}'.format(i)
                    )(ypred)
                elif loss_criteria=='poisson':
                    ypred = tfp.layers.DistributionLambda(
                        lambda t: tfp.distributions.Poisson(log_rate=t[..., 0]),
                        name='task{}'.format(i)
                    )(ypred)
                ypreds.append(ypred)
            model = tf.keras.Model(inputs=x, outputs=ypreds)
        elif self.loss_criteria=='zero-inflated-poisson':
            if architecture=='shared':
                leaf_dims = (2*self.num_tasks, )
                submodel = models_multitask.create_multitask_submodel(
                    x,
                    self.num_trees,
                    self.depth,
                    2*self.num_tasks,
                    leaf_dims,
                    "MultitaskRegressionClassification",
                    model_type=self.model_type,
                    alpha=self.alpha,
                    power=self.power,
                )
                x = submodel.input
                outputs = submodel(x)
                regression_preds,  classification_logodds= tf.split(outputs, num_or_size_splits=2, axis=1)
                regression_preds = tf.keras.layers.Activation('linear', name="MultitaskRegression")(regression_preds)
                classification_logodds = tf.keras.layers.Activation('linear', name="MultitaskClassification")(classification_logodds)
                regression_preds = tf.split(regression_preds, num_or_size_splits=self.num_tasks, axis=1)
                classification_logodds = tf.split(classification_logodds, num_or_size_splits=self.num_tasks, axis=1)

            elif architecture=='shared-per-task':
                leaf_dims = (2*self.num_tasks, )
                submodel = models_multitask.create_multitask_submodel(
                    x,
                    self.num_trees,
                    self.depth,
                    self.num_tasks,
                    leaf_dims,
                    "MultitaskRegressionClassification",
                    model_type=self.model_type,
                    alpha=self.alpha,
                    power=self.power,
                )
                x = submodel.input
                outputs = submodel(x)
                task_preds = tf.split(outputs, num_or_size_splits=self.num_tasks, axis=1)
                regression_preds = []
                classification_logodds = []
                for i, task_pred in enumerate(task_preds):
                    regression_pred,  classification_logodd= tf.split(task_pred, num_or_size_splits=2, axis=1)
                    regression_pred = tf.keras.layers.Activation('linear', name="MultitaskRegression-task{}".format(i))(regression_pred)
                    classification_logodd = tf.keras.layers.Activation('linear', name="MultitaskClassification-task{}".format(i))(classification_logodd)
                    regression_preds.append(regression_pred)
                    classification_logodds.append(classification_logodd)                

            ypreds = []
            for i, (rp, cp) in enumerate(zip(regression_preds, classification_logodds)):
                ypred = tf.keras.layers.Concatenate(axis=1)([rp, cp])
                ypred = tfp.layers.DistributionLambda(layers.zero_inflation_mixture_singletask, name='task{}'.format(i))(ypred)
                ypreds.append(ypred)

            model = tf.keras.Model(inputs=x, outputs=ypreds)
        elif self.loss_criteria in ['negative-binomial']:
            if self.architecture == 'shared': 
                leaf_dims = (2*self.num_tasks, )
                submodel = models_multitask.create_multitask_submodel(
                    x,
                    self.num_trees,
                    self.depth,
                    2*self.num_tasks,
                    leaf_dims,
                    "MultitaskRegressionDispersion",
                    model_type=self.model_type,
                    alpha=self.alpha,
                    power=self.power,
                )
                x = submodel.input
                outputs = submodel(x)
                regression_preds,  dispersion_preds= tf.split(outputs, num_or_size_splits=2, axis=1)
                regression_preds = tf.keras.layers.Activation('linear', name="MultitaskRegression")(regression_preds)
                dispersion_preds = tf.keras.layers.Activation('linear', name="MultitaskDispersion")(dispersion_preds)
                regression_preds = tf.split(regression_preds, num_or_size_splits=self.num_tasks, axis=1)
                dispersion_preds = tf.split(dispersion_preds, num_or_size_splits=self.num_tasks, axis=1)
                ypreds = []
                for i, (rp, dp) in enumerate(zip(regression_preds, dispersion_preds)):
                    ypred = tf.keras.layers.Concatenate(axis=1)([rp, dp])
                    ypred = tfp.layers.DistributionLambda(
                        lambda t: tfp.distributions.NegativeBinomial.experimental_from_mean_dispersion(
                            mean=tf.math.exp(t[..., 0]), dispersion=tf.math.exp(t[..., 1])
                        ),
                        name='task{}'.format(i)
                    )(ypred)
                    ypreds.append(ypred)
                
            elif self.architecture == 'shared-per-task': 
                leaf_dims = (2*self.num_tasks, )
                submodel = models_multitask.create_multitask_submodel(
                    x,
                    self.num_trees,
                    self.depth,
                    self.num_tasks,
                    leaf_dims,
                    "MultitaskRegression",
                    model_type=self.model_type,
                    alpha=self.alpha,
                    power=self.power,
                )
                x = submodel.input
                outputs = submodel(x)
                outputs = tf.split(outputs, num_or_size_splits=self.num_tasks, axis=1)
                ypreds = []
                for i, rp in enumerate(outputs):
                    ypred = tf.keras.layers.Activation('linear')(rp)
                    ypred = tfp.layers.DistributionLambda(
                        lambda t: tfp.distributions.NegativeBinomial.experimental_from_mean_dispersion(
                            mean=tf.math.exp(t[..., 0]), dispersion=tf.math.exp(t[..., 1])
                        ),
                        name='task{}'.format(i)
                    )(ypred)
                    ypreds.append(ypred)
            model = tf.keras.Model(inputs=x, outputs=ypreds)

        # model.summary()
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model
    
    def process_data(
        self,
        y_true,
        w
    ):
        y_true_processed = deepcopy(y_true)
        w_masked = ~np.isnan(y_true_processed)*w.reshape(-1,1)
        y_true_processed[np.isnan(y_true_processed)] = self.dummy_response
        y_true_processed = y_true_processed.astype('float32')
        w_masked = w_masked.astype('float32')
        w_masked_dict = {}
        for i in range(self.num_tasks):
            w_masked_dict['task{}'.format(i)]=w_masked[:,i]*self.task_weights[i]    
        return y_true_processed, w_masked_dict
    
    def train(
        self,
        x_train,
        y_train,
        w_train,
        x_valid,
        y_valid,
        w_valid,
    ):
        # print(y_valid_pred_dist)
        y_train_processed, w_train_dict = self.process_data(y_train, w_train)
        y_valid_processed, w_valid_dict = self.process_data(y_valid, w_valid)
        
        callbacks = [
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=25, verbose=1, mode='auto', restore_best_weights=True)
        ]
        if len(get_available_gpus())==0:
            history = self.model.fit(x=x_train, 
                      y=tf.split(y_train_processed, num_or_size_splits=self.num_tasks, axis=1),
                      sample_weight=w_train_dict,
                      epochs=self.epochs, 
                      batch_size=self.batch_size, 
                      shuffle=True,
                      callbacks=callbacks,
                      validation_data=(
                          x_valid,
                          tf.split(y_valid_processed, num_or_size_splits=self.num_tasks, axis=1),
                          w_valid_dict
                      ),
                      verbose=1, 
                      ) 
            
        else:
            with tf.device(get_available_gpus()[0]):
                history = self.model.fit(x=x_train, 
                          y=tf.split(y_train_processed, num_or_size_splits=self.num_tasks, axis=1),
                          sample_weight=w_train_dict,
                          epochs=self.epochs, 
                          batch_size=self.batch_size, 
                          shuffle=True,
                          callbacks=callbacks,
                          validation_data=(
                              x_valid,
                              tf.split(y_valid_processed, num_or_size_splits=self.num_tasks, axis=1),
                              w_valid_dict
                          ),
                          verbose=1, 
                          ) 
    
    def predict_distribution(self, x):
        with tf.device(get_available_cpus()[0]):
            y_pred_dist = self.model(x) 
        return y_pred_dist
            
    def predict(self, x):
        y_pred_dist = self.predict_distribution(x)
        y_pred = tf.stack([yv.mean() for yv in y_pred_dist], axis=1).numpy() # compute expectation
        return y_pred
    
    def compute_metrics(
        self,
        x,
        y_true,
        w,
    ):
        y_true_processed, w_dict = self.process_data(y_true, w)
        w_masked = np.array([v for k,v in w_dict.items()]).T 
        y_pred_dist = self.predict_distribution(x)
        negloglikelihood = [
            losses.NegativeLogLikelihood()(
                yv, yvp, sw
            ).numpy() for yv, yvp, sw in zip(
                np.array_split(y_true_processed, self.num_tasks, axis=1),
                y_pred_dist,
                np.array_split(w_masked, self.num_tasks, axis=1)
            )
        ]
        y_pred = self.predict(x)
        mse = utils_multitask.multitask_masked_mean_squared_error(y_true, y_pred, w)
        poisson_deviance, _, _ = utils_multitask.multitask_masked_poisson_deviance(y_true, y_pred, w)
        metrics = {
            'negloglikelihood': negloglikelihood,
            'mse': mse,
            'poisson_deviance': poisson_deviance,
        }
        return metrics

    def evaluate(
        self,
        x,
        y_true,
        w,                
    ):
        metrics = self.compute_metrics(x, y_true, w)
        return metrics

