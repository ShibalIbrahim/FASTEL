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