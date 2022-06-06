import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras import activations
from tensorflow.keras import regularizers
from tensorflow.keras import constraints

def SmoothStep(gamma):
    def smooth_step(x):
        cond1 = tf.cast(tf.math.less_equal(x, -gamma/2), tf.float32)
        cond2 = tf.cast(tf.math.logical_and(tf.math.greater_equal(x, -gamma/2), tf.math.less_equal(x, gamma/2)), tf.float32)
        cond3 = tf.cast(tf.math.greater_equal(x, gamma/2), tf.float32)

        a = tf.math.multiply(cond1, 0.0)
        b = tf.math.multiply(cond2, (-2/(gamma**3))*tf.math.pow(x, 3) + (3/(2*gamma))*x + 0.5)
        c = tf.math.multiply(cond3, 1.0)
        
        f = a + b + c
        return f
    return smooth_step

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

def projection_onto_simplex_batch(w):    
    batch = tf.shape(w)[0]
    n = tf.shape(w)[1]
    y = tf.sort(w, direction='DESCENDING', axis=1)
    thresh = (tf.cumsum(y, axis=1)-1.0)/tf.expand_dims(tf.cast(n-tf.range(n-1,-1,-1), dtype=w.dtype), axis=0)
    
    i = tf.searchsorted(thresh-y, tf.expand_dims(tf.zeros(batch, dtype=w.dtype), axis=1))
    t = tf.gather(thresh, tf.math.minimum(i,n)-1, axis=1, batch_dims=1)
    w = tf.maximum(w-t,0)
    return w

class SparseProjectionOntoSimplexBatch(tf.keras.constraints.Constraint):
    """Projects weights onto a simplex after gradient update in a batch-wise fashion.
    
    Solves the following minimization problem:
        min (1/2)*||y-w||^2 s.t. \sum_i w_i = 1, w_i>=0, ||w||_0<=k
    
    References:
        [Sparse projections onto the simplex](https://arxiv.org/pdf/1206.1529.pdf) 
        by Anastasios Kyrillidis, Stephen Becker, Volkan Cevher, Christoph Koch in ICML
    Returns:
        w: Float Tensor of shape (p,).
    """
    def __init__(self, k=1, anneal=False, steps=1000, **kwargs):
        super(SparseProjectionOntoSimplexBatch, self).__init__(**kwargs)
        self.k = k
        self.anneal = anneal
        self.steps = steps
        self.iterations = tf.Variable(initial_value=0, trainable=False, name='iterations')
            
    def __call__(self, w):
        self.iterations.assign_add(1)
        p = tf.shape(w)[1]
        if self.anneal:
            # temp1 = tf.cast(self.iterations, dtype=w.dtype)/tf.cast(self.steps, dtype=w.dtype)
            # temp2 = tf.clip_by_value(
            #     temp1,
            #     k = tf.minimum(k, 2), 
            #     tf.cast(p, dtype=w.dtype)-1.
            # )
            # k = p - tf.cast(tf.math.round(temp2), dtype=tf.int32)
            
            k = tf.where(
                tf.greater(self.iterations, tf.constant(10)),
                tf.constant(1),
                tf.minimum(p, tf.constant(2))
            )
            # tf.print("=========steps:", self.iterations, "k:", k)
        else:
            k = self.k
            
        topk = tf.math.top_k(w, k)
        w_topk = projection_onto_simplex_batch(tf.gather(w, topk.indices, axis=1, batch_dims=1))        
        num_rows = tf.shape(w)[0]
        row_range = tf.range(num_rows)
        row_tensor = tf.tile(row_range[:,None], (1, k))
        topk_row_col_indices = tf.stack([row_tensor, topk.indices], axis=2)
        y = tf.scatter_nd(topk_row_col_indices, w_topk, tf.shape(w))
        return y
    
    def get_config(self):
        config = super(SparseProjectionOntoSimplexBatch, self).get_config()
        return config

def count_trees_per_depth(model):
    num_trees_per_depth = np.sum(model.layers[1].layers[1].get_weights()[0]>0., axis=0)
    k = np.max(np.sum(model.layers[1].layers[1].get_weights()[0]>0., axis=1))
    return k, num_trees_per_depth

# Callback class to save training loss and the number of trees at each depth
class TreePerDepthHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        k, trees_per_depth = count_trees_per_depth(self.model)
        self.k = [k]
        self.trees_per_depth = [trees_per_depth]

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        k, trees_per_depth = count_trees_per_depth(self.model)
        self.trees_per_depth.append(trees_per_depth)
        self.k.append(k)


class TreeEnsembleWithDepthPruning(tf.keras.layers.Layer):    
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
                 kernel_regularizer=tf.keras.regularizers.l2(0.0),
                 complexity_regularization=0.,
                 anneal=False,
                 steps=100,
                 mixture_of_experts_type='sparse-simplex'):
        super(TreeEnsembleWithDepthPruning, self).__init__()
        self.max_depth = max_depth
        self.leaf_dims = leaf_dims
        self.num_trees = num_trees
        self.complexity_regularization = complexity_regularization
        self.mixture_of_experts_type = mixture_of_experts_type
        self.anneal = anneal
        self.steps = steps
        
        self.depth_experts = [
            TreeEnsemble(
                num_trees,
                depth,
                leaf_dims,
                activation=activation,
                kernel_regularizer=kernel_regularizer,
                combine_output=False
            ) for depth in range(1, max_depth+1)
        ]
        cost_matrix = np.triu(np.ones((self.max_depth, self.max_depth)))
        self.cost_matrix = tf.cast(cost_matrix, dtype=self.dtype)


    def build(self, input_shape):
        self.expert_weights = self.add_weight(
            shape=[self.num_trees, self.max_depth],
            trainable=True,
            constraint=SparseProjectionOntoSimplexBatch(k=1, anneal=self.anneal, steps=self.steps),
            name="expert_weights"
        )
        
    def call(self, input):
        outputs = [expert(input) for expert in self.depth_experts]
        outputs = [tf.expand_dims(o, axis=3) for o in outputs]
        outputs = tf.concat(outputs, axis=3)
        # tf.print("===============expert_weights:", self.expert_weights)
        output = tf.tensordot(outputs, self.expert_weights, axes=[[2,3],[0,1]]) 
        cost_complexity = tf.reduce_sum(tf.tensordot(self.cost_matrix, self.expert_weights, axes=[[1],[1]])) / tf.cast(self.num_trees, dtype=self.dtype)
        self.add_loss(self.complexity_regularization*cost_complexity)
        # self.add_metric(tf.reduce_sum(self.expert_weights, axis=0), name="d-trees")
        return output            
        
    def get_config(self):
        config = super(TreeEnsembleWithDepthPruning, self).get_config()
        config.update({"num_trees": self.num_trees})
        config.update({"max_depth": self.max_depth})
        config.update({"leaf_dims": self.leaf_dims})
        config.update({"node_index": self.node_index})
        config.update({"activation": self.activation})
        config.update({"kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer)})
        config.update({"complexity_regularization": self.complexity_regularization})
        return config


class SharedTreeEnsembleWithDepthPruning(tf.keras.layers.Layer):
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
                 kernel_regularizer=tf.keras.regularizers.L2(0.),
                 complexity_regularization=0.,
                 anneal=False,
                 steps=100,
                 mixture_of_experts_type='sparse-simplex'):
        super(SharedTreeEnsembleWithDepthPruning, self).__init__()
        self.max_depth = max_depth
        self.leaf_dims = leaf_dims
        self.num_trees = num_trees
        self.node_index = node_index
        self.leaf = node_index >= 2**max_depth - 1
        self.internal_eps = internal_eps
        self.kernel_regularizer = kernel_regularizer
        self.complexity_regularization = complexity_regularization
        self.mixture_of_experts_type = mixture_of_experts_type
        self.anneal = anneal
        self.steps = steps
        if not self.leaf:
            self.dense_layer = tf.keras.layers.Dense(
                self.num_trees,
                kernel_regularizer=self.kernel_regularizer,
                activation='sigmoid',
            )
            self.left_child = SharedTreeEnsembleWithDepthPruning(
                self.num_trees,
                self.max_depth,
                self.leaf_dims,
                activation=activation,
                node_index=2*self.node_index+1,
                kernel_regularizer=self.kernel_regularizer
            )
            self.right_child = SharedTreeEnsembleWithDepthPruning(
                self.num_trees,
                self.max_depth,
                self.leaf_dims,
                activation=activation,
                node_index=2*self.node_index+2,
                kernel_regularizer=self.kernel_regularizer
            )  
        if self.node_index==0:
            cost_matrix = np.triu(np.ones((self.max_depth, self.max_depth)))
            self.cost_matrix = tf.cast(cost_matrix, dtype=self.dtype)


    def build(self, input_shape):
        if self.node_index==0:
            self.expert_weights = self.add_weight(
                shape=[self.num_trees, self.max_depth],
                trainable=True,
                constraint=SparseProjectionOntoSimplexBatch(k=1, anneal=self.anneal, steps=self.steps),
                name="expert_weights-"+str(self.node_index)
            )
        else:
            self.leaf_weight = self.add_weight(
                shape=[1, self.leaf_dims, self.num_trees],
                trainable=True,
                name="Node-{}-Leaf".format(self.node_index))
            
        
    def call(self, input, prob=1.0):
        if self.node_index>0:
            if not self.leaf:
                # shape = (batch_size, num_trees)
                current_prob = tf.keras.backend.clip(self.dense_layer(input), self.internal_eps, 1-self.internal_eps)
                expectations_left_child = self.left_child(input, current_prob * prob)
                expectations_right_child = self.right_child(input, (1 - current_prob) * prob)
                expectations = [exp_left_child + exp_right_child for exp_left_child, exp_right_child in zip(expectations_left_child, expectations_right_child)]

                expectation_self = tf.expand_dims(prob, axis=1) * self.leaf_weight
                expectations = [expectation_self] + expectations
            else:
                # self.leaf_weight's shape = (1, self.leaf_dims, num_trees)
                # prob's shape = (batch_size, num_trees)
                expectation = tf.expand_dims(prob, axis=1) * self.leaf_weight
                expectations = [expectation]
            return expectations
        else:
            current_prob = tf.keras.backend.clip(self.dense_layer(input), self.internal_eps, 1-self.internal_eps)
            expectations_left_child = self.left_child(input, current_prob * prob)
            expectations_right_child = self.right_child(input, (1 - current_prob) * prob)
            expectations = [exp_left_child + exp_right_child for exp_left_child, exp_right_child in zip(expectations_left_child, expectations_right_child)]

            expectations = [tf.expand_dims(exp, axis=3) for exp in expectations]
            expectations = tf.concat(expectations, axis=3)
            # tf.print("===============expert_weights:", self.expert_weights)
            output = tf.tensordot(expectations, self.expert_weights, axes=[[2,3],[0,1]]) 
            cost_complexity = tf.reduce_sum(tf.tensordot(self.cost_matrix, self.expert_weights, axes=[[1],[1]])) / tf.cast(self.num_trees, dtype=self.dtype)
            self.add_loss(self.complexity_regularization*cost_complexity)
            # self.add_metric(tf.reduce_sum(self.expert_weights, axis=0), aggregation=None, name="d-trees")
            return output            
        
    def get_config(self):
        config = super(SharedTreeEnsembleWithDepthPruning, self).get_config()
        config.update({"num_trees": self.num_trees})
        config.update({"max_depth": self.max_depth})
        config.update({"leaf_dims": self.leaf_dims})
        config.update({"activation": self.activation})
        config.update({"node_index": self.node_index})
        config.update({"kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer)})
        config.update({"complexity_regularization": self.complexity_regularization})
        return config
    
    
# class TreeEnsembleTensor(tf.keras.layers.Layer):
#     """An ensemble of soft decision trees.
    
#     The layer returns the sum of the decision trees in the ensemble.
#     Each soft tree returns a vector, whose dimension is specified using
#     the `leaf_dims' parameter.
    
#     Implementation Notes:
#         This is a fully vectorized implementation. It treats the ensemble
#         as one "super" tree, where every node stores a dense layer with 
#         num_trees units, each corresponding to the hyperplane of one tree.
    
#     Input:
#         An input tensor of shape = (batch_size, ...)

#     Output:
#         An output tensor of shape = (batch_size, leaf_dims)
#     """

#     def __init__(self, num_trees, max_depth, leaf_dims, node_index=0, kernel_regularizer=tf.keras.regularizers.l2(0.0)):
#         super(TreeEnsembleTensor, self).__init__()
#         self.max_depth = max_depth
#         self.leaf_dims = leaf_dims
#         self.num_trees = num_trees
#         self.node_index = node_index
#         self.leaf = node_index >= 2**max_depth - 1
#         if not self.leaf:
#             self.dense_layer = tf.keras.layers.Dense(self.num_trees, activation='sigmoid')
#             self.left_child = TreeEnsembleTensor(self.num_trees, self.max_depth, self.leaf_dims, 2*self.node_index+1)
#             self.right_child = TreeEnsembleTensor(self.num_trees, self.max_depth, self.leaf_dims, 2*self.node_index+2)

#     def build(self, input_shape):
#         if self.leaf:
#             self.leaf_weight = self.add_weight(shape=[1, self.leaf_dims, self.num_trees], trainable=True, name="Node-"+str(self.node_index))
        
#     def call(self, input, prob=1.0):
#         if not self.leaf:
#             # shape = (batch_size, num_trees)
#             current_prob = self.dense_layer(input)
#             return self.left_child(input, current_prob * prob) + self.right_child(input, (1 - current_prob) * prob)
#         else:
#             # self.leaf_weight's shape = (1, self.leaf_dims, num_trees)
#             # prob's shape = (batch_size, num_trees)
#             return tf.expand_dims(prob, axis=1) * self.leaf_weight
        
#     def get_config(self):
#         config = super(TreeEnsembleTensor, self).get_config()
#         config.update({"num_trees": self.num_trees})
#         config.update({"max_depth": self.max_depth})
#         config.update({"leaf_dims": self.leaf_dims})
#         config.update({"node_index": self.node_index})
#         return config    
    
    
class MixtureOfExpertsTreeEnsemble(tf.keras.layers.Layer):
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

    def __init__(self, num_trees, max_depth, leaf_dims, node_index=0, mixture_of_experts_type='softmax'):
        super(MixtureOfExpertsTreeEnsemble, self).__init__()
        self.max_depth = max_depth
        self.leaf_dims = leaf_dims
        self.num_trees = num_trees
        self.node_index = node_index
        self.leaf = node_index >= 2**max_depth - 1
        self.mixture_of_experts_type = mixture_of_experts_type
        if not self.leaf:
            self.dense_layer = tf.keras.layers.Dense(self.num_trees, activation='sigmoid')
            self.left_child = MixtureOfExpertsTreeEnsemble(
                self.num_trees,
                self.max_depth,
                self.leaf_dims,
                2*self.node_index+1,
                mixture_of_experts_type=self.mixture_of_experts_type
            )
            self.right_child = MixtureOfExpertsTreeEnsemble(
                self.num_trees,
                self.max_depth,
                self.leaf_dims,
                2*self.node_index+2,
                mixture_of_experts_type=self.mixture_of_experts_type
            )

    def build(self, input_shape):
        if self.leaf:
            self.leaf_weight = self.add_weight(shape=[1, self.leaf_dims, self.num_trees], trainable=True, name="Node-"+str(self.node_index))
            if self.mixture_of_experts_type == 'softmax':
                self.gates = self.add_weight(
                    shape=[1, self.leaf_dims, self.num_trees],
                    trainable=True,
                    name="Gate-"+str(self.node_index)
                )
                self.gates_activated = tf.keras.activations.softmax(self.gates, axis=-1)
        
    def call(self, input, prob=1.0):
        if not self.leaf:
            # shape = (batch_size, num_trees)
            current_prob = self.dense_layer(input)
            return self.left_child(input, current_prob * prob) + self.right_child(input, (1 - current_prob) * prob)
        else:
            # self.leaf_weight's shape = (1, self.leaf_dims, num_trees)
            # prob's shape = (batch_size, num_trees)
            output = tf.expand_dims(prob, axis=1) * self.leaf_weight
            if self.mixture_of_experts_type == 'softmax':
                output *= self.gates_activated
            return tf.math.reduce_sum(output, axis=2)
        
    def get_config(self):
        config = super(MixtureOfExpertsTreeEnsemble, self).get_config()
        config.update({"num_trees": self.num_trees})
        config.update({"max_depth": self.max_depth})
        config.update({"leaf_dims": self.leaf_dims})
        config.update({"node_index": self.node_index})
        config.update({"mixture_of_experts_type": self.mixture_of_experts_type})
        return config
    
# Old implementation.
# class TreeEnsemble(tf.keras.layers.Layer):
#     """An ensemble of soft decision trees.
    
#     The layer returns the sum of the decision trees in the ensemble.
#     Each soft tree returns a vector, whose dimension is specified using
#     the `leaf_dims' parameter.
    
#     Input:
#         An input tensor of shape = (batch_size, ...)

#     Output:
#         An output tensor of shape = (batch_size, leaf_dims)
#     """

#     def __init__(self, num_trees, depth, leaf_dims=1):
#         """Constructor for TreeEnsemble.

#         Args:
#             num_trees: Number of trees in the ensemble.
#             depth: Depth of each tree. Note: in the current implementation,
#                 all trees are fully grown to depth.
#             leaf_dims: Dimension of the leaf outputs. The output of the ensemble
#                 for a single sample is a vector with leaf_dims entries.
#         """
#         super(TreeEnsemble, self).__init__()
#         self.num_trees = num_trees
#         self.depth = depth
#         self.leaf_dims = leaf_dims
#         self.single_trees = []
#         for _ in range(self.num_trees):
#             self.single_trees.append(SoftTreeVectorized(self.depth, self.leaf_dims))

#     def call(self, input):
#         return tf.math.accumulate_n([self.single_trees[tree_index](input) for tree_index in range(self.num_trees)])

#     def get_config(self):
#         config = super(TreeEnsemble, self).get_config()
#         config.update({"num_trees": self.num_trees})
#         config.update({"depth": self.depth})
#         config.update({"leaf_dims": self.leaf_dims})
#         return config

class SoftTree(tf.keras.layers.Layer):
    """A soft decision tree."""

    def __init__(self, depth, leaf_dims=1):
        super(SoftTree, self).__init__()
        self.depth = depth
        self.leaf_dims = leaf_dims
        self.num_nodes = 2**(self.depth+1) - 1
        self.num_internal_nodes = 2**(self.depth) - 1
        self.nodes_dense_layers = []
        for _ in range(self.num_internal_nodes):
            self.nodes_dense_layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))
        
    def build(self, input_shape):
        self.leaves_weights = self.add_weight(name="leaves_weights", shape=[self.num_nodes - self.num_internal_nodes, self.leaf_dims], trainable=True)    

    def call(self, input):
        # Each element is a vector, where every entry is the probability
        # of a sample reaching the current node.
        self.node_probabilities = [None for _ in range(self.num_nodes)]
        self.node_probabilities[0] = 1.0 # tf.ones_like(self.nodes_dense_layers[0](input))
        # queue is initialized with the index of the root node.
        queue = [0]
        while len(queue) != 0:
            current_node_index = queue.pop(0)
            # Compute the probability vector at the current node.
            current_prob_vector = self.nodes_dense_layers[current_node_index](input)
            # Generate the children.
            left_child_index = 2*current_node_index + 1
            self.node_probabilities[left_child_index] = current_prob_vector * self.node_probabilities[current_node_index]
            right_child_index = 2*current_node_index + 2
            self.node_probabilities[right_child_index] = (1 - current_prob_vector) * self.node_probabilities[current_node_index]

            # Push internal nodes to the queue.
            if left_child_index < self.num_internal_nodes:
                queue.append(left_child_index)
                queue.append(right_child_index)

        outputs = []
        for i in range(self.num_internal_nodes, self.num_nodes):
            output = tf.expand_dims(self.leaves_weights[i - self.num_internal_nodes, :], axis=0) * self.node_probabilities[i]
            outputs.append(output)
        return tf.math.accumulate_n(outputs)
    
    def get_config(self):
        config = super(SoftTree, self).get_config()
        config.update({"depth": self.depth})
        config.update({"leaf_dims": self.leaf_dims})
        return config
    
    
class SoftTreeVectorized(tf.keras.layers.Layer):
    """A soft decision tree."""
    def __init__(self, depth, leaf_dims=1):
        super(SoftTreeVectorized, self).__init__()
        self.depth = depth
        self.leaf_dims = leaf_dims
        self.num_nodes = 2**(self.depth+1) - 1
        self.num_internal_nodes = 2**(self.depth) - 1
        self.nodes_dense_layers = tf.keras.layers.Dense(self.num_internal_nodes, activation='sigmoid')
        
    def build(self, input_shape):
        self.leaves_weights = self.add_weight(name="leaves_weights", shape=[self.num_nodes - self.num_internal_nodes, self.leaf_dims], trainable=True)    

    @tf.function
    def call(self, input):
        # Each element is a vector, where every entry is the probability
        # of a sample reaching the current node.
        self.node_probabilities = [None for _ in range(self.num_nodes)]
        self.node_probabilities[0] = 1.0 # tf.ones_like(self.nodes_dense_layers[0](input))
        # queue is initialized with the index of the root node.
        queue = [0]
        prob_vectors = self.nodes_dense_layers(input)
        while len(queue) != 0:
            current_node_index = queue.pop(0)
            # Compute the probability vector at the current node.
            current_prob_vector = tf.expand_dims(prob_vectors[:,current_node_index], axis=-1) #self.nodes_dense_layers[current_node_index](input)
            # Generate the children.
            left_child_index = 2*current_node_index + 1
            self.node_probabilities[left_child_index] = current_prob_vector * self.node_probabilities[current_node_index]
            right_child_index = 2*current_node_index + 2
            self.node_probabilities[right_child_index] = (1 - current_prob_vector) * self.node_probabilities[current_node_index]

            # Push internal nodes to the queue.
            if left_child_index < self.num_internal_nodes:
                queue.append(left_child_index)
                queue.append(right_child_index)

        outputs = []
        for i in range(self.num_internal_nodes, self.num_nodes):
            output = tf.expand_dims(self.leaves_weights[i - self.num_internal_nodes, :], axis=0) * self.node_probabilities[i]
            outputs.append(output)
        return tf.math.accumulate_n(outputs)
    
    def get_config(self):
        config = super(SoftTreeVectorized, self).get_config()
        config.update({"depth": self.depth})
        config.update({"leaf_dims": self.leaf_dims})
        return config
