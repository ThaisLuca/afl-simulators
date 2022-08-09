
from collections import OrderedDict
from typing import List, Tuple, Optional

import data
import flwr as fl
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from flwr.server import ServerConfig
from flwr.common.logger import log
from client import FlowerClient
from model import Net

from strategy import ClusterStrategy
from utils import *

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load dataset (CIFAR-10)
trainloaders, valloaders, testloader = data.Dataset().load_datasets()

def fit_round(rnd: int):
    """Send round number to client."""
    return {"rnd": rnd}

def evaluate_config(rnd: int):
  """Return evaluation configuration dict for each round.
  Perform five local evaluation steps on each client (i.e., use five
  batches) during rounds one to three, then increase to ten local
  evaluation steps.
  """
  val_steps = 5 if rnd < 4 else 10
  return {"val_steps": val_steps}

def get_eval_fn(model):
  """Return an evaluation function for server-side evaluation."""

  # The `evaluate` function will be called after every round
  def evaluate(server_round, parameters, config):
    set_parameters(model, parameters) # Update model with the latest parameters
    loss, accuracy = test(model, testloader)
    return float(loss), {"accuracy": float(accuracy)}
  return evaluate

# def get_eval_fn(model):
#   """Return an evaluation function for server-side evaluation."""

#   # The `evaluate` function will be called after every round
#   def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
#     set_parameters(model, weights) # Update model with the latest parameters
#     loss, accuracy = test(model, servertestloader)
#     return float(loss), {"accuracy": float(accuracy)}
#   return evaluate

def client_fn(cid: str) -> FlowerClient:

    # Create model
    model = Net().to(DEVICE)

    # Load data (CIFAR-10)
    #trainloaders, valloaders, testloader = data.load_datasets()

    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(cid, model, trainloader, valloader)

# Start Flower server for NUM_ROUNDS rounds of federated learning
if __name__ == "__main__":

    # Creates server's models
    # models = []
    # for i in range(N_CLUSTERS):
    #     models.append(Net().to(DEVICE))

    model = Net().to(DEVICE)

    #fl.server.strategy.FedAvg
    strategy = ClusterStrategy(min_available_clients=MIN_AVAILABLE_CLIENTS,
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=FRACTION_EVAL,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVAL_CLIENTS,
        n_clusters=N_CLUSTERS,
        on_evaluate_config_fn=evaluate_config,
        evaluate_fn=get_eval_fn(model),
        #evaluate_metrics_aggregation_fn=agg
        #on_fit_config_fn=fit_round,
        #fraction_fit=FRACTION_FIT,
        #fraction_eval=FRACTION_EVAL,
        #min_fit_clients=MIN_FIT_CLIENTS,
        #min_eval_clients=MIN_EVAL_CLIENTS,
    )

    # Start simulation
    fl.simulation.start_simulation(config=ServerConfig(num_rounds=NUM_ROUNDS),
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        strategy=strategy
    )