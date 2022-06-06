"""Losses for multitask learning with missing responses with sample weights
"""

import tensorflow as tf

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

