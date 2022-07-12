

import os
import math
import utils
import numpy as np
from typing import Dict
from client import MnistClient
from sklearn.metrics import log_loss
from strategy import HalfOfWeightsStrategy
from sklearn.linear_model import LogisticRegression

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl
import tensorflow as tf

NUM_CLIENTS = 10
NUM_ROUNDS = 3
MIN_AVAILABLE_CLIENTS = int(NUM_CLIENTS * 0.75)   # Wait until at least 75 clients are available
FRACTION_FIT = 0.1                                # Sample 10% of available clients for training
MIN_FIT_CLIENTS = 10                              # Never sample less than 10 clients for training
FRACTION_EVAL = 0.05                              # Sample 5% of available clients for evaluation
MIN_EVAL_CLIENTS = 5                              # Never sample less than 5 clients for evaluation

def fit_round(rnd: int) -> Dict:
    """Send round number to client."""
    return {"rnd": rnd}


def get_eval_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    _, (X_test, y_test) = utils.load_mnist()

    def evaluate(parameters: fl.common.Weights):
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}
    return evaluate


def client_fn(cid: str) -> fl.client.Client:

    # Create model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load data partition (divide MNIST into NUM_CLIENTS distinct partitions)
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    partition_size = math.floor(len(x_train) / NUM_CLIENTS)
    idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size
    x_train_cid = x_train[idx_from:idx_to] / 255.0
    y_train_cid = y_train[idx_from:idx_to]

    # Use 10% of the client's training data for validation
    split_idx = math.floor(len(x_train) * 0.9)
    x_train_cid, y_train_cid = x_train_cid[:split_idx], y_train_cid[:split_idx]
    x_val_cid, y_val_cid = x_train_cid[split_idx:], y_train_cid[split_idx:]

    # Create and return client
    return MnistClient(model, x_train_cid, y_train_cid, x_val_cid, y_val_cid)

# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = HalfOfWeightsStrategy(fraction_fit=FRACTION_FIT,
        fraction_eval=FRACTION_EVAL,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_eval_clients=MIN_EVAL_CLIENTS,
        min_available_clients=MIN_AVAILABLE_CLIENTS
    )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        num_rounds=NUM_ROUNDS,
        strategy=strategy,
    )