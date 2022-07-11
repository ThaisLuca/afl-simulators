
import utils
import flwr as fl
import numpy as np
from typing import Dict
from client import MnistClient
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

NUM_CLIENTS = 3
NUM_ROUNDS = 5
MIN_AVAILABLE_CLIENTS = 2

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

    # Load data partition (divide MNIST into NUM_CLIENTS distinct partitions)
    (X_train, y_train), (X_test, y_test) = utils.load_mnist()

    partition_id = np.random.choice(10)
    (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

    # Create Model
    model = LogisticRegression(
    penalty="l2",
    max_iter=1,  # local epoch
    warm_start=True,  # prevent refreshing weights when fitting
    )

    utils.set_initial_params(model)

    # Create and return client
    return MnistClient(model, X_train, y_train)

# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
    )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        num_rounds=NUM_ROUNDS,
        strategy=strategy,
    )