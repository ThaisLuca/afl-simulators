import warnings
import flwr as fl
import numpy as np
import random

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils

class MnistClient(fl.client.NumPyClient):
    def __init__(self):
      self._ID = random.randint(100,200)
      print(f"""Starting Client {self._ID}""")

    def get_parameters(self):  # type: ignore
        return utils.get_model_parameters(model)

    def fit(self, parameters, config):  # type: ignore
        utils.set_model_params(model, parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        print(f"Client {self._ID}: Training finished for round {config['rnd']}")
        results = {
            "loss": log_loss(y_train, model.predict_proba(X_train)),
            "accuracy": model.score(X_train, y_train),
        }
        params = utils.get_model_parameters(model)
        return params, len(X_train), results

    def evaluate(self, parameters, config):  # type: ignore
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, len(X_test), {"accuracy": accuracy}

if __name__ == "__main__":

    (X_train, y_train), (X_test, y_test) = utils.load_mnist()

    partition_id = np.random.choice(10)
    (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

    model = LogisticRegression(
    penalty="l2",
    max_iter=1,  # local epoch
    warm_start=True,  # prevent refreshing weights when fitting
    )

    utils.set_initial_params(model)

    fl.client.start_numpy_client("127.0.0.1:4466", client=MnistClient())