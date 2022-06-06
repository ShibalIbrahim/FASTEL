import sys
# sys.argv = sys.argv[:1]  This command is needed to run tensorflow tests in jupyter notebook to address flags

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
import tensorflow_probability as tfp
tfd = tfp.distributions

class MultitaskLossesTest(tf.test.TestCase):
    def test_block_distributions(self):
        
        d = tfd.Blockwise(
            [
                tfd.Independent(
                    tfd.Normal(
                        loc=tf.zeros(4, dtype=tf.float64),
                        scale=1),
                    reinterpreted_batch_ndims=1),
                tfd.MultivariateNormalTriL(
                    scale_tril=tf.eye(2, dtype=tf.float32)),
            ],
            dtype_override=tf.float32,
        )
        print(d)
        x = d.sample([2, 1])
        y = d.log_prob(x)
        x.shape  # ==> (2, 1, 4 + 2)
        x.dtype  # ==> tf.float32
        y.shape  # ==> (2, 1)
        y.dtype  # ==> tf.float32

        print(d.mean()) 
                
if __name__ == '__main__':
    tf.test.main()