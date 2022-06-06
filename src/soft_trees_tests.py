import tensorflow as tf
from soft_trees import TreeEnsemble, MultitaskTreeEnsemble
from soft_trees import TreeEnsembleWithDepthPruning, SharedTreeEnsembleWithDepthPruning
from soft_trees import count_trees_per_depth, TreePerDepthHistory

# class TreeEnsembleTest(tf.test.TestCase):
#     def test_training(self):
#         mnist = tf.keras.datasets.mnist
#         (x_train, y_train), (x_test, y_test) = mnist.load_data()
#         x_train, x_test = x_train / 255.0, x_test / 255.0
#         # Create a tree ensemble with 5 trees, each with depth 3 and 10 entries in the leaves.
#         # The tree ensemble will be used as the output layer ==> Each entry in the output vector
#         # corresponds to one of the 10 digits.
#         tree_ensemble = TreeEnsemble(5, 3, 10)
#         model = tf.keras.models.Sequential([
#           tf.keras.layers.Flatten(input_shape=(28, 28)),
#           tree_ensemble
#         ])
#         loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#         model.compile(optimizer='adam',
#                       loss=loss_fn,
#                       metrics=['accuracy'])
#         model.summary()
#         for l in model.layers:
#             print("===============================================", l.name)
#             for w in l.get_weights():
#                 print(w.shape)
#         history = model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))
#         # Check if the accuracy is above 0.9.
#         self.assertGreaterEqual(history.history['accuracy'][-1], 0.9)

        
# class MultitaskTreeEnsembleTest(tf.test.TestCase):
#     def test_training(self):
#         mnist = tf.keras.datasets.mnist
#         (x_train, y_train), (x_test, y_test) = mnist.load_data()
#         x_train, x_test = x_train / 255.0, x_test / 255.0
#         # Create a tree ensemble with 5 trees, each with depth 3 and 10 entries in the leaves.
#         # The tree ensemble will be used as the output layer ==> Each entry in the output vector
#         # corresponds to one of the 10 digits.
#         tree_ensemble = MultitaskTreeEnsemble(5, 3, 10)
#         model = tf.keras.models.Sequential([
#           tf.keras.layers.Flatten(input_shape=(28, 28)),
#           tree_ensemble
#         ])
#         loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#         model.compile(optimizer='adam',
#                       loss=loss_fn,
#                       metrics=['accuracy'])
#         history = model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))
#         # Check if the accuracy is above 0.9.
#         self.assertGreaterEqual(history.history['accuracy'][-1], 0.9)
#         model.save("model_test")

class TreeEnsembleWithDepthPruningTest(tf.test.TestCase):
    def test_tree_ensemble_with_depth_pruning(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        # Create a tree ensemble with 5 trees, each with depth 3 and 10 entries in the leaves.
        # The tree ensemble will be used as the output layer ==> Each entry in the output vector
        # corresponds to one of the 10 digits.
        lr=5.0
        tree_ensemble = TreeEnsembleWithDepthPruning(
            5,
            3,
            10,
            kernel_regularizer=None,
            complexity_regularization=2.5e-1,
        )
        model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tree_ensemble
        ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=tf.keras.optimizers.SGD(lr),
                      loss=loss_fn,
                      metrics=['accuracy'])
        model.summary()
        for l in model.layers:
            print("===============================================", l.name)
            for w in l.get_weights():
                print(w.shape)
        td = TreePerDepthHistory()
        callbacks = [td]
        history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=callbacks)
        print("=========td:", td.trees_per_depth)
        # Check if the accuracy is above 0.9.
        self.assertGreaterEqual(history.history['accuracy'][-1], 0.9)

    def test_shared_tree_ensemble_with_depth_pruning(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        # Create a tree ensemble with 5 trees, each with depth 3 and 10 entries in the leaves.
        # The tree ensemble will be used as the output layer ==> Each entry in the output vector
        # corresponds to one of the 10 digits.
        lr=5.0
        tree_ensemble = SharedTreeEnsembleWithDepthPruning(
            5,
            3,
            10,
            kernel_regularizer=None,
            complexity_regularization=2.5e-1,
        )
        model = tf.keras.models.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tree_ensemble
        ])
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer=tf.keras.optimizers.SGD(lr),
                      loss=loss_fn,
                      metrics=['accuracy'])
        model.summary()
        for l in model.layers:
            print("===============================================", l.name)
            for w in l.get_weights():
                print(w.shape)
        td = TreePerDepthHistory()
        callbacks = [td]
        history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), callbacks=callbacks)
        print("=========td:", td.trees_per_depth)                
        # Check if the accuracy is above 0.9.
        self.assertGreaterEqual(history.history['accuracy'][-1], 0.9)
        
tf.test.main()