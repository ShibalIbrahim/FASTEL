"""Creates models for multitask learning with Neural Networks.
"""
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from soft_trees import TreeEnsemble, MultitaskTreeEnsemble

############################# Multitask Models with Differentiable Decision Trees ############################
def create_multitask_submodel(x,
                              num_trees,
                              depth,
                              num_tasks,
                              leaf_dims,
                              task_name,
                              activation='sigmoid',
                              model_type=None,
                              kernel_regularizer=tf.keras.regularizers.L2(0.0),
                              alpha=1.0,
                              power=1.0):
    """Creates a submodel for a task with soft decision trees.
    
    Args:
      x: Keras Input instance.
      num_layers: integer, scalar.
      num_trees: Number of trees in the ensemble, int scalar.
      depth: Depth of each tree. Note: in the current implementation,
        all trees are fully grown to depth, int scalar.
      leaf_dims: list of dimensions of leaf outputs for each ensemble layer,
        int tuple of shape (num_layers, ).
      activation: 'sigmoid' or 'smooth-step'
      task_name: name of submodel, string.
      
    Returns:
      Keras submodel instantiation
    """
    
    y = x
    if model_type is None:
        y = TreeEnsemble(
            num_trees,
            depth,
            leaf_dims[-1],
            activation=activation,
            kernel_regularizer=kernel_regularizer
        )(y) 
    elif model_type=='regularized':
        y = MultitaskTreeEnsemble(
            num_trees,
            depth,
            num_tasks,
            leaf_dims[-1],
            activation=activation,
            alpha=alpha,
            power=power,
            # kernel_regularizer=kernel_regularizer
        )(y) # output layer
        
    submodel = models.Model(inputs=x, outputs=y, name=task_name)
    return submodel