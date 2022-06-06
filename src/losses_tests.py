import sys
# sys.argv = sys.argv[:1]  This command is needed to run tensorflow tests in jupyter notebook to address flags

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
import losses

class MultitaskLossesTest(tf.test.TestCase):
    def test_multitask_masked_mean_squared_error(self):
        y_true = constant_op.constant([1, np.nan, 2,
                                       np.nan, -2, 6],
                                      shape=(2, 3),
                                      dtype=dtypes.float32)
        y_pred = constant_op.constant([4, 8, 12,
                                       8, 1, 3],
                                      shape=(2, 3),
                                      dtype=dtypes.float32)
        task_weights = np.ones(y_true.shape[1])
        mse_obj = losses.MultitaskMaskedMeanSquaredError(task_weights)
        sample_weight = constant_op.constant([1.2, 3.4], shape=(2, ),
                                              dtype=dtypes.float32)
        loss = mse_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 96.0)
        
    def test_multitask_masked_poisson_deviance(self):
        y_true = constant_op.constant([1, np.nan, 2,
                                       np.nan, 0, 6],
                                      shape=(2, 3),
                                      dtype=dtypes.float32)
        y_pred = constant_op.constant([4, 8, 12,
                                       8, 1, 3],
                                      shape=(2, 3),
                                      dtype=dtypes.float32)
        task_weights = np.ones(y_true.shape[1])
        pd_obj = losses.MultitaskMaskedPoissonDeviance(task_weights)
        sample_weight = constant_op.constant([1.2, 3.4], shape=(2, ),
                                              dtype=dtypes.float32)
        loss = pd_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss.numpy(), 8.4882105, delta=1e-5)
        
if __name__ == '__main__':
    tf.test.main()