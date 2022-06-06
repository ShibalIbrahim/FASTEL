"""Losses for multitask learning with missing responses with sample weights
"""

import tensorflow as tf

class MultitaskMaskedMeanSquaredError(tf.keras.losses.Loss):
    """Multitask Mean Squared Error Loss."""
    def __init__(self,
                 task_weights,
                 **kwargs):
        super(MultitaskMaskedMeanSquaredError, self).__init__(
            # reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name='MultitaskMaskedMeanSquaredError',
            **kwargs)
        self.task_weights = tf.expand_dims(tf.constant(task_weights, dtype=tf.float64), axis=0)
    
    def call(self, y_true, y_pred):
        assert y_true.dtype==y_pred.dtype, "dtypes for y_true:{} and y_pred:{} do not match.".format(y_true.dtype, y_pred.dtype)
        mask = tf.math.logical_not(tf.math.is_nan(y_true))        
        loss = tf.math.multiply_no_nan(
             tf.math.squared_difference(y_true, y_pred),
             tf.cast(mask, dtype=y_true.dtype)
         )
        loss = tf.reduce_sum(
            tf.multiply(
                loss,
                tf.cast(self.task_weights, dtype=y_true.dtype)
                ),
            axis=1)
        return loss
    
    def get_config(self):
        config = super(MultitaskMaskedMeanSquaredError, self).get_config()
        config.update({'reduction': self.reduction,
                       'name': self.name,
                       'task_weights': self.task_weights.numpy()})
        return config

class MultitaskMaskedMeanSquaredErrorSampleWeights(tf.keras.losses.Loss):
    """Multitask Mean Squared Error Loss with Multitask Sample Weights."""
    def __init__(self,
                 task_weights,
                 **kwargs):
        super(MultitaskMaskedMeanSquaredErrorSampleWeights, self).__init__(
            #reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name='MultitaskMaskedMeanSquaredErrorSampleWeights',
            **kwargs)
        self.task_weights = tf.expand_dims(tf.constant(task_weights, dtype=tf.float64), axis=0)
    
    def call(self, y_true, y_pred):
        y_pred, sample_weights = tf.split(y_pred, num_or_size_splits=2, axis=1)
        assert y_true.dtype==y_pred.dtype, "dtypes for y_true:{} and y_pred:{} do not match.".format(y_true.dtype, y_pred.dtype)
        weights = tf.math.multiply(
            tf.cast(sample_weights, dtype=y_true.dtype),
            tf.cast(self.task_weights, dtype=y_true.dtype)
        )
        loss = tf.math.multiply(
             tf.math.squared_difference(y_true, y_pred),
             weights
        )
        loss = tf.reduce_sum(loss, axis=1)
        return loss
    
    def get_config(self):
        config = super(MultitaskMaskedMeanSquaredErrorSampleWeights, self).get_config()
        config.update({'reduction': self.reduction,
                       'name': self.name,
                       'task_weights': self.task_weights.numpy()})
        return config

    
class MultitaskMaskedPoissonError(tf.keras.losses.Loss):
    """Multitask Poisson Deviance Loss."""
    def __init__(self,
                 task_weights,
                 eps=1e-6,
                 **kwargs):
        super(MultitaskMaskedPoissonError, self).__init__(
            # reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name='MultitaskMaskedPoissonError',
            **kwargs)
        self.task_weights = tf.expand_dims(tf.constant(task_weights, dtype=tf.float64), axis=0)
        self.eps = tf.constant(eps, dtype=tf.float64)

    def compute_poisson_error(self, y_t, y_p):
        poisson_dev = y_p - tf.multiply(
                y_t,
                tf.math.log(y_p + tf.cast(self.eps, dtype=y_t.dtype)*tf.ones_like(y_p)) # handles y_p = 0. 
            )
        return poisson_dev

    def call(self, y_true, y_pred):
        assert y_true.dtype==y_pred.dtype, "dtypes for y_true:{} and y_pred:{} do not match.".format(y_true.dtype, y_pred.dtype)
        mask = tf.math.logical_not(tf.math.is_nan(y_true))
        loss = tf.math.multiply_no_nan(
             self.compute_poisson_error(y_true, y_pred),
             tf.cast(mask, dtype=y_true.dtype)
         )
        loss = tf.reduce_sum(
            tf.multiply(
                loss,
                tf.cast(self.task_weights, dtype=y_true.dtype)
                ),
            axis=1)
        return loss
    
    def get_config(self):
        config = super(MultitaskMaskedPoissonError, self).get_config()
        config.update({'reduction': self.reduction,
                       'name': self.name,
                       'task_weights': self.task_weights.numpy()})
        return config

class MultitaskMaskedPoissonErrorSampleWeights(tf.keras.losses.Loss):
    """Multitask Poisson Loss with Multitask Sample Weights."""
    def __init__(self,
                 task_weights,
                 eps=1e-6,
                 **kwargs):
        super(MultitaskMaskedPoissonErrorSampleWeights, self).__init__(
            # reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
            name='MultitaskMaskedPoissonErrorSampleWeights',
            **kwargs)
        self.task_weights = tf.expand_dims(tf.constant(task_weights, dtype=tf.float64), axis=0)
        self.eps = tf.constant(eps, dtype=tf.float64)
    
    def compute_poisson_error(self, y_t, y_p):
        poisson_dev = y_p - tf.multiply(
                y_t,
                tf.math.log(y_p + tf.cast(self.eps, dtype=y_t.dtype)*tf.ones_like(y_p)) # handles y_p = 0. 
            )
        return poisson_dev

    def call(self, y_true, y_pred):
        y_pred, sample_weights = tf.split(y_pred, num_or_size_splits=2, axis=1)
        assert y_true.dtype==y_pred.dtype, "dtypes for y_true:{} and y_pred:{} do not match.".format(y_true.dtype, y_pred.dtype)
        weights = tf.math.multiply(
            tf.cast(sample_weights, dtype=y_true.dtype),
            tf.cast(self.task_weights, dtype=y_true.dtype)
        )
        loss = tf.math.multiply(
             self.compute_poisson_error(y_true, y_pred),
             weights
         )
        loss = tf.reduce_sum(loss, axis=1)
        return loss
    
    def get_config(self):
        config = super(MultitaskMaskedPoissonErrorSampleWeights, self).get_config()
        config.update({'reduction': self.reduction,
                       'name': self.name,
                       'task_weights': self.task_weights.numpy(),
                       'eps': self.eps.numpy()})
        return config

    
class NegativeLogLikelihood(tf.keras.losses.Loss):
    """Multitask Negative LogLikelihood Loss."""
    def __init__(self,
                 reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 name='NegativeLogLikelihood',
                 **kwargs):
        super(NegativeLogLikelihood, self).__init__(
            **kwargs)
        pass
     
    def negative_log_likelihood(self, y_t, y_p):
        nll = -y_p.log_prob(tf.reshape(y_t,(-1,)))
        return nll
  
    def call(self, y_true, y_pred):
#         assert y_true.dtype==y_pred.dtype, "dtypes for y_true:{} and y_pred:{} do not match.".format(y_true.dtype, y_pred.dtype)
        loss = self.negative_log_likelihood(y_true, y_pred)
        return loss
    
    def get_config(self):
        config = super(NegativeLogLikelihood, self).get_config()
        config.update({'reduction': self.reduction,
                       'name': self.name})
        return config

    
# class MultitaskMaskedNegativeLogLikelihoodSampleWeights(tf.keras.losses.Loss):
#     """Multitask Negative LogLikelihood Loss with Multitask Sample Weights."""
#     def __init__(self,
#                  task_weights,
#                  **kwargs):
#         super(MultitaskMaskedNegativeLogLikelihoodSampleWeights, self).__init__(
#             # reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
#             name='MultitaskMaskedNegativeLogLikelihoodSampleWeights',
#             **kwargs)
#         self.task_weights = tf.expand_dims(tf.constant(task_weights, dtype=tf.float64), axis=0)
     
#     def negative_log_likelihood(self, y_t, y_p):
#         print(y_t)
#         print(y_p)
#         nll = -y_p.log_prob(y_t)
# #         nll = -y_p.log_prob(tf.reshape(y_t,(-1,)))
#         return nll
  
#     def call(self, y_true, y_pred):
#         print(y_pred.get_shape())
#         sample_weights = y_pred[1]
#         y_pred = y_pred[0]
# #         y_pred, sample_weights = tf.split(y_pred, num_or_size_splits=2, axis=1)
#         print(y_pred)
#         print(sample_weights)    
#         assert y_true.dtype==y_pred.dtype, "dtypes for y_true:{} and y_pred:{} do not match.".format(y_true.dtype, y_pred.dtype)
#         weights = tf.math.multiply(
#             tf.cast(sample_weights, dtype=y_true.dtype),
#             tf.cast(self.task_weights, dtype=y_true.dtype)
#         )
#         loss = tf.math.multiply(
#              self.negative_log_likelihood(y_true, y_pred),
#              weights
#          )
#         loss = tf.reduce_sum(loss, axis=1)
#         return loss
    
#     def get_config(self):
#         config = super(MultitaskMaskedNegativeLogLikelihoodSampleWeights, self).get_config()
#         config.update({'reduction': self.reduction,
#                        'name': self.name,
#                        'task_weights': self.task_weights.numpy()})
#         return config


class MultitaskMaskedPoissonDeviance(tf.keras.losses.Loss):
    """Multitask Poisson Deviance Loss."""
    def __init__(self,
                 task_weights,
                 eps=1e-6,
                 **kwargs):
        super(MultitaskMaskedPoissonDeviance, self).__init__(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                                            name='MultitaskMaskedPoissonDeviance',
                                            **kwargs)
        self.task_weights = tf.expand_dims(tf.constant(task_weights, dtype=tf.float64), axis=0)
        self.eps = tf.constant(eps, dtype=tf.float64)

    def compute_poisson_deviance(self, y_t, y_p):
        poisson_dev = tf.multiply(
                y_t,
                tf.math.log(tf.divide(y_t, y_p + tf.cast(self.eps, dtype=y_t.dtype)*tf.ones_like(y_p))) # handles y_p = 0. 
            ) - y_t + y_p
        return poisson_dev

    def call(self, y_true, y_pred):
        assert y_true.dtype==y_pred.dtype, "dtypes for y_true:{} and y_pred:{} do not match.".format(y_true.dtype, y_pred.dtype)
        mask = tf.math.logical_not(tf.math.is_nan(y_true))
        loss = tf.math.multiply_no_nan(
             tf.where(
                tf.math.equal(y_true, tf.zeros_like(y_true)), 
                -y_true + y_pred,
                self.compute_poisson_deviance(y_true, y_pred) # handles y_true != 0.
             ),
             tf.cast(mask, dtype=y_true.dtype)
         )
        loss = tf.reduce_sum(
            tf.multiply(
                loss,
                tf.cast(self.task_weights, dtype=y_true.dtype)
                ),
            axis=1)
        print_op = tf.print("Loss:", loss)
        with tf.control_dependencies([print_op]):
            return loss
    
    def get_config(self):
        config = super(MultitaskMaskedPoissonDeviance, self).get_config()
        config.update({'reduction': self.reduction,
                       'name': self.name,
                       'task_weights': self.task_weights.numpy()})
        return config


# ###### TODO(shibal): Update for use as loss and add a test
# def poisson_deviance_metric(y_true, y_pred, sample_weight=None):
#     """Calculates Poisson Deviance.
    
#     Args:
#       y_true: actual task responses, a float Tensor of shape (num_samples, num_tasks).
#       y_pred: predicted task responses, a float Tensor of shape (num_samples, num_tasks).
#       sample_weight: sample weights for each sample per task
#         - None, (default translates to 1 for all).
#         - a float Tensor of shape (num_samples, num_tasks), allows handling of missing samples.
    
#     Returns:
#       poisson deviance, a float scalar.
#     """
#     eps = 1e-6
#     def compute_poisson_deviance(y_t, y_p, s_w):
#         poisson_dev = tf.multiply(
#             tf.multiply(
#                 y_t,
#                 tf.math.log(tf.divide(y_t, y_p + eps)) # handles y_pred = 0. 
#             ) - y_t + y_p,
#             s_w
#         )
#         return poisson_dev
    
#     assert y_true.dtype==y_pred.dtype 
#     if sample_weight is None:   
#         sample_weight = tf.ones((tf.shape(y_true)[0], ), dtype=y_true.dtype)
        
#     poisson_deviances = []
#     for y_t, y_p in zip(tf.unstack(y_true, axis=1), tf.unstack(y_pred, axis=1)):
#         mask = tf.math.logical_not(tf.math.is_nan(y_true))
#         y_t = y_t[mask]
#         y_p = y_p[mask]
#         s_w = sample_weight[mask]
#         poisson_deviances.append(
#             tf.divide(
#                 tf.reduce_sum(
#                     tf.where(
#                         y_t==0.0, 
#                         tf.multiply(-y_t + y_p, s_w),
#                         compute_poisson_deviance(y_t, y_p, s_w) # handles y_true != 0.
#                     )
#                 ),
#                 tf.reduce_sum(s_w)
#             )        
#         )
#     poisson_deviance = tf.reduce_sum(tf.stack(poisson_deviances))
#     return poisson_deviance