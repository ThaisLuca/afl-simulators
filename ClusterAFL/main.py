

import os
import math
import numpy as np
from typing import Dict
from client import FlowerClient
from sklearn.metrics import log_loss
from strategy import HalfOfWeightsStrategy,ClusterStrategy
from sklearn.linear_model import LogisticRegression

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl
import tensorflow as tf

NUM_CLIENTS = 100
NUM_ROUNDS = 3
MIN_AVAILABLE_CLIENTS = 2 # int(NUM_CLIENTS * 0.75)# Wait until at least 75 clients are available
#FRACTION_FIT = 0.1                                # Sample 10% of available clients for training
#MIN_FIT_CLIENTS = 10                              # Never sample less than 10 clients for training
#FRACTION_EVAL = 0.05                              # Sample 5% of available clients for evaluation
#MIN_EVAL_CLIENTS = 5                              # Never sample less than 5 clients for evaluation

def fit_round(rnd: int) -> Dict:
    """Send round number to client."""
    return {"rnd": rnd}

def client_fn(cid: str) -> fl.client.Client:

    # Create model
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load data partition (divide MNIST into NUM_CLIENTS distinct partitions)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Create and return client
    return FlowerClient(cid, model, x_train[:1000], y_train[:1000], x_test[1000:2000], y_test[1000:2000])

# Start Flower server for NUM_ROUNDS rounds of federated learning
if __name__ == "__main__":
    #fl.server.strategy.FedAvg
    strategy = fl.server.strategy.FedAvg(min_available_clients=MIN_AVAILABLE_CLIENTS,
        #on_fit_config_fn=fit_round,
        #fraction_fit=FRACTION_FIT,
        #fraction_eval=FRACTION_EVAL,
        #min_fit_clients=MIN_FIT_CLIENTS,
        #min_eval_clients=MIN_EVAL_CLIENTS,
    )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        num_rounds=NUM_ROUNDS,
        strategy=strategy,
    )