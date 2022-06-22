import warnings
import flwr as fl
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import utils


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