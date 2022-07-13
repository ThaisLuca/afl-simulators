
import warnings
import flwr as fl
import numpy as np
import tensorflow as tf

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, ID, model, x_train, y_train, x_test, y_test) -> None:
        super().__init__()
        self.ID = ID
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        return self.model.get_weights()

    @tf.function(experimental_relax_shapes=True) 
    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, steps_per_epoch=3, verbose=2)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test,verbose=2)
        return loss, len(self.x_test), {"accuracy": accuracy}