"""Recommendataion on WSDM Dataset.

Author: DHSong
Last Modified At: 2020.07.07

Recommend data on WSDM Dataset.
"""

from tensorflow import keras

class NeuralNetModel(keras.Model):
    def __init__(self):
        super(NeuralNetModel, self).__init__()

        self.dense1 = keras.layers.Dense(
            32, 
            activation='relu', 
            kernel_initializer='glorot_normal', 
            name='dense1'
        )
        self.dense2 = keras.layers.Dense(
            1, 
            activation='sigmoid', 
            kernel_initializer='glorot_normal', 
            name='dense2'
        )

    def call(self, x):
        layer = self.dense1(x)
        output = self.dense2(layer)
        return output 
