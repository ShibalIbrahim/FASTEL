import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import activations
from tensorflow.keras import regularizers
from tensorflow.keras import constraints


class SmoothStep(tf.keras.layers.Layer):
    """A smooth-step function.
    For a scalar x, the smooth-step function is defined as follows:
    0                                             if x <= -gamma/2
    1                                             if x >= gamma/2
    3*x/(2*gamma) -2*x*x*x/(gamma**3) + 0.5       o.w.
    See https://arxiv.org/abs/2002.07772 for more details on this function.
    """

    def __init__(self, gamma=1.0):
        """Initializes the layer.
        Args:
          gamma: Scaling parameter controlling the width of the polynomial region.
        """
        super(SmoothStep, self).__init__()
        self._lower_bound = -gamma / 2
        self._upper_bound = gamma / 2
        self._a3 = -2 / (gamma**3)
        self._a1 = 3 / (2 * gamma)
        self._a0 = 0.5

    def call(self, inputs):
        return tf.where(
            inputs <= self._lower_bound,
            tf.zeros_like(inputs),
            tf.where(
                inputs >= self._upper_bound,
                tf.ones_like(inputs),
                self._a3 * (inputs**3) + self._a1 * inputs + self._a0
            )
        )

class TreeEnsemble(tf.keras.layers.Layer):
    """An ensemble of soft decision trees.
    
    The layer returns the sum of the decision trees in the ensemble.
    Each soft tree returns a vector, whose dimension is specified using
    the `leaf_dims' parameter.
    
    Implementation Notes:
        This is a fully vectorized implementation. It treats the ensemble
        as one "super" tree, where every node stores a dense layer with 
        num_trees units, each corresponding to the hyperplane of one tree.
    
    Input:
        An input tensor of shape = (batch_size, ...)

    Output:
        An output tensor of shape = (batch_size, leaf_dims)
    """

    def __init__(self,
                 num_trees,
                 max_depth,
                 leaf_dims,
                 activation='sigmoid',
                 node_index=0,
                 internal_eps=0,
                 kernel_regularizer=tf.keras.regularizers.L2(0.0),
                 combine_output=True):
        super(TreeEnsemble, self).__init__()
        self.max_depth = max_depth
        self.leaf_dims = leaf_dims
        self.num_trees = num_trees
        self.node_index = node_index
        self.leaf = node_index >= 2**max_depth - 1
        self.internal_eps = internal_eps
        self.kernel_regularizer = kernel_regularizer
        self.combine_output = combine_output
        if not self.leaf:
            self.dense_layer = tf.keras.layers.Dense(
                self.num_trees,
                kernel_regularizer=self.kernel_regularizer,
                activation='sigmoid',
            )
            self.left_child = TreeEnsemble(
                self.num_trees,
                self.max_depth,
                self.leaf_dims,
                activation=activation,
                node_index=2*self.node_index+1,
                kernel_regularizer=self.kernel_regularizer,
                combine_output=combine_output
            )
            self.right_child = TreeEnsemble(
                self.num_trees,
                self.max_depth,
                self.leaf_dims,
                activation=activation,
                node_index=2*self.node_index+2,
                kernel_regularizer=self.kernel_regularizer,
                combine_output=combine_output
            )

    def build(self, input_shape):
        if self.leaf:
            self.leaf_weight = self.add_weight(
                shape=[1, self.leaf_dims, self.num_trees],
                trainable=True,
                name="Node-"+str(self.node_index))
        
    def call(self, input, prob=1.0):
        if not self.leaf:
            # shape = (batch_size, num_trees)
            current_prob = tf.keras.backend.clip(self.dense_layer(input), self.internal_eps, 1-self.internal_eps)
            return self.left_child(input, current_prob * prob) + self.right_child(input, (1 - current_prob) * prob)
        else:
            # self.leaf_weight's shape = (1, self.leaf_dims, num_trees)
            # prob's shape = (batch_size, num_trees)
            output = tf.expand_dims(prob, axis=1) * self.leaf_weight
            if self.combine_output:
                output = tf.math.reduce_sum(output, axis=2)
            return output
        
    def get_config(self):
        config = super(TreeEnsemble, self).get_config()
        config.update({"num_trees": self.num_trees})
        config.update({"max_depth": self.max_depth})
        config.update({"leaf_dims": self.leaf_dims})
        config.update({"activation": self.activation})
        config.update({"node_index": self.node_index})
        config.update({"kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer)})
        config.update({"combine_output": self.combine_output})        
        return config    
    
class LinearRegularized(tf.keras.layers.Layer):
    """Regularized Split SuperNode across tasks.
    
    The layer returns the conditional probability at the split node for each task.
    
    Implementation Notes:
        This is a fully vectorized implementation of the split node for multitask tree ensemlble.
        The supernode is a dense layer with (num_task, num_trees) units, each corresponding to a
        hyperplane of one tree of the ensemble of one task.
    
    Input:
        An input tensor of shape = (batch_size, num_features)

    Output:
        An output tensor of shape = (batch_size, units)
    """
    def __init__(self,
                 units,
                 alpha=1.0,
                 activation='sigmoid',
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                ):
        super(LinearRegularized, self).__init__()
        self.units = units
        self.alpha = tf.cast(alpha, dtype=self.dtype)
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.use_bias = use_bias

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], *self.units),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        if self.use_bias:
            self.b = self.add_weight(
                shape=(*self.units,),
                initializer=self.bias_initializer,
                trainable=True,
            )

    def pairwise_dist(self, A):
        """Write an algorithm that computes batched the 2-norm distance between each pair of two collections of row vectors.
        We use the euclidean distance metric.
        For a matrix A [m, d] and a matrix B [n, d] we expect a matrix of pairwise distances here D [m, n]
        # Arguments:
            A: A tf.Tensor object. The first matrix.
            B: A tf.tensor object. The second matrix.
        # Returns:
            Calculate distance.
        # Reference:
            [scipy.spatial.distance.cdist](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
            [tensorflow/tensorflow#30659](https://github.com/tensorflow/tensorflow/issues/30659)
        """

        # squared norms of each row in A and B
        na = tf.reduce_sum(tf.square(A), 1)

        # na as a row and nb as a column vectors
        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(na, [1, -1])

        # return pairwise euclidean difference matrix
        D = tf.maximum(na - 2 * tf.matmul(A, A, False, True) + nb, 0.0)
        return D

    def pairwise_dist_v2(self, A):
        """Write an algorithm that computes batched the 2-norm distance. Uses less memory allocation
        We use the euclidean distance metric.
        For a matrix A [p, T, m] and a matrix B [p, T, m] we expect a matrix of pairwise distances here D [T, T]

        # Arguments:
            A: A tf.Tensor object. The first matrix.
            B: A tf.tensor object. The second matrix.
        # Returns:
            Calculate distance.
        """

        # squared norms of each row in A and B
        na = tf.reduce_sum(tf.square(A), axis=[0,2])

        # na as a row and nb as a column vectors
        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(na, [1, -1])

        # return pairwise euclidean difference matrix tf.matmul(A, A, False, True)
        D = tf.maximum(na - 2 * tf.tensordot(A, A, axes=[[0,2],[0,2]]) + nb, 0.0)
        return D
    
    def call(self, inputs):
        output = tf.tensordot(inputs, self.w, axes=[[-1], [0]]) 
        if self.use_bias:
            output += self.b
        loss = tf.reduce_mean(
            # self.pairwise_dist(tf.reshape(tf.transpose(self.w, perm=[1,0,2]), [self.units[0], -1]))
            self.pairwise_dist_v2(self.w)
        )
        self.add_loss(self.alpha*loss)
        return self.activation(output)

    def get_config(self):
        config = super(LinearRegularized, self).get_config()
        config.update({"units": self.units})
        config.update({"activation": activations.serialize(self.activation)})
        config.update({"kernel_initializer": initializers.serialize(self.kernel_initializer)})
        config.update({"bias_initializer": initializers.serialize(self.bias_initializer)})
        return config


class MultitaskTreeEnsemble(tf.keras.layers.Layer):
    """An ensemble of soft decision trees.
    
    The layer returns the sum of the decision trees in the ensemble.
    Each soft tree returns a vector, whose dimension is specified using
    the `leaf_dims' parameter.
    
    Implementation Notes:
        This is a fully vectorized implementation. It treats the ensemble
        as one "super" tree, where every node stores a dense layer with 
        num_trees units, each corresponding to the hyperplane of one tree.
    
    Input:
        An input tensor of shape = (batch_size, num_features)

    Output:
        An output tensor of shape = (batch_size, leaf_dims)
    """

    def __init__(self,
                 num_trees,
                 max_depth,
                 num_tasks,
                 leaf_dims,
                 activation='sigmoid',
                 node_index=0,
                 depth_index=0,
                 internal_eps=0,
                 alpha=1.0,
                 power=2.0,
                 name='Node-Root',
                 **kwargs):
        super(MultitaskTreeEnsemble, self).__init__(name=name,**kwargs)
        self.max_depth = max_depth
        self.num_tasks = num_tasks
        self.leaf_dims = leaf_dims
        self.task_size = (int)(leaf_dims/num_tasks) 
        self.num_trees = num_trees
        self.activation = activation
        self.node_index = node_index
        self.depth_index = depth_index
        self.leaf = node_index >= 2**max_depth - 1
        self.internal_eps = internal_eps
        self.alpha = alpha
        self.power = power
        if not self.leaf:
            self.dense_layer = LinearRegularized(
                (self.num_tasks, self.num_trees),
                alpha=self.alpha/(self.power**self.depth_index),
                activation=activation,
            )
            self.left_child = MultitaskTreeEnsemble(
                self.num_trees,
                self.max_depth,
                self.num_tasks,
                self.leaf_dims,
                activation=activation,
                node_index=2*self.node_index+1,
                depth_index=self.depth_index+1,
                alpha=self.alpha,
                power=self.power,
                name="Node-"+str(2*self.node_index+1)
            )
            self.right_child = MultitaskTreeEnsemble(
                self.num_trees,
                self.max_depth,
                self.num_tasks,
                self.leaf_dims,
                activation=activation,
                node_index=2*self.node_index+2,
                depth_index=self.depth_index+1,
                alpha=self.alpha,
                power=self.power,
                name="Node-"+str(2*self.node_index+2)
            )

    def build(self, input_shape):
        if self.leaf:
            self.leaf_weight = self.add_weight(shape=[1, self.leaf_dims, self.num_trees], trainable=True, name="Node-"+str(self.node_index))
        
    def call(self, input, prob=1.0):
        if not self.leaf:
            # shape = (batch_size, num_trees)
            current_prob = tf.keras.backend.clip(self.dense_layer(input), self.internal_eps, 1-self.internal_eps)
            return self.left_child(input, current_prob * prob) + self.right_child(input, (1 - current_prob) * prob)
        else:
            # self.leaf_weight's shape = (1, self.leaf_dims, num_trees)
            # prob's shape = (batch_size, num_trees)
            return tf.math.reduce_sum(tf.tile(prob, [1, self.task_size, 1]) * self.leaf_weight, axis=2)
    def get_config(self):
        config = super(MultitaskTreeEnsemble, self).get_config()
        config.update({"num_trees": self.num_trees})
        config.update({"max_depth": self.max_depth})
        config.update({"leaf_dims": self.leaf_dims})
        config.update({"activation": self.activation})
        config.update({"node_index": self.node_index})
        config.update({"depth_index": self.depth_index})
        config.update({"alpha": self.alpha})
        config.update({"power": self.power})
        return config