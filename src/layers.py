import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.keras.initializers import RandomUniform
from tensorflow.python.keras.layers import Layer, InputSpec

######################## Zero-inflation Mixture Distribution Layers ######################
def zero_inflation_mixture_singletask(y_pred): 
    log_preds, log_odds = tf.split(y_pred, num_or_size_splits=2, axis=1) # regression, classification
    
    num_tasks = log_preds.get_shape()[1]
    assert num_tasks==1

    log_preds = tf.squeeze(log_preds)

    # Codes for the zero inflation, using sigmoid squeezes the value between 0 and 1.
    s = tf.math.sigmoid(log_odds) 

    # The two probabilities for zeros or Poissonian
    probs = tf.concat([1-s, s], axis=1)

    # Create a mixture of two components.
    distribution = tfd.Mixture(
        cat=tfd.Categorical(probs=probs), # tfd.Categorical allows to create a mixture of two components. 
        components=[
            tfd.Deterministic(loc=tf.zeros_like(log_preds)), # Zero as a deterministic value 
            tfd.Poisson(log_rate=log_preds), # Value drawn from a Poissonian
        ]
    )
    return distribution

# ZeroInflatedPoisson = tfp.layers.DistributionLambda(zero_inflation_mixture_singletask)

def zero_inflation_negative_binomial_mixture_singletask(y_pred): 
    log_preds_mean, unnormalized_pred_dispersion, log_odds = tf.split(y_pred, num_or_size_splits=3, axis=1) # regression, classification
    
    num_tasks = log_preds_mean.get_shape()[1]
    assert num_tasks==1

    log_preds_mean = tf.squeeze(log_preds_mean)
    unnormalized_pred_dispersion = tf.squeeze(unnormalized_pred_dispersion)

    # Negative Binomial Exponential is used to guarantee values for mean >0.
    # Unit 1: Outputs log(mean) of the negative binomial ==> Need to apply exp() to get the mean. 
    # Unit 2: Outputs the (un-normalized) dispersion of the negative binomial ==> Need to apply sigmoid() to get the dispersion.
    mean = tf.math.exp(log_preds_mean)
    dispersion = tf.math.sigmoid(unnormalized_pred_dispersion)

    # Codes for the zero inflation, using sigmoid squeezes the value between 0 and 1.
    s = tf.math.sigmoid(log_odds) 

    # The two probabilities for zeros or Negative Binomial
    probs = tf.concat([1-s, s], axis=1)

    # Create a mixture of two components.
    distribution = tfd.Mixture(
        cat=tfd.Categorical(probs=probs), # tfd.Categorical allows to create a mixture of two components. 
        components=[
            tfd.Deterministic(loc=tf.zeros_like(mean)), # Zero as a deterministic value 
            tfd.NegativeBinomial.experimental_from_mean_dispersion(mean=mean, dispersion=dispersion), # Value drawn from a Negative Binomial
        ]
    )
    return distribution