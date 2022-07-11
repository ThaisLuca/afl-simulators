
import warnings
import flwr as fl
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils

class MnistClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train) -> None:
        self.model = model
        self.X_train, self.y_train = X_train, y_train

    def get_parameters(self):  # type: ignore
        return utils.get_model_parameters(self.model)

    def fit(self, parameters, config):  # type: ignore
        utils.set_model_params(self.model, parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.X_train, self.y_train)
        print(f"Training finished for round {config['rnd']}")
        return utils.get_model_parameters(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        utils.set_model_params(self.model, parameters)
        loss = log_loss(self.y_test, self.model.predict_proba(X_test))
        accuracy = self.model.score(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}