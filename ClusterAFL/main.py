
from collections import OrderedDict
from typing import List, Tuple, Optional

import flwr as fl
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

from flwr.common.logger import log
from client import FlowerClient
from model import Net
from utils import *

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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

  # Download and transform CIFAR-10 (train and test)
  transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
  )
  testset = CIFAR10("./dataset", train=False, download=True, transform=transform)
  testloader = DataLoader(testset)

  # The `evaluate` function will be called after every round
  def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
    set_parameters(model, weights) # Update model with the latest parameters
    loss, accuracy = test(model, testloader)
    return float(loss), {"accuracy": float(accuracy)}
  return evaluate

def load_datasets():
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
      [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)

    # Split training set into 10 partitions to simulate the individual dataset
    partition_size = len(trainset) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders, valloaders, testloader


def client_fn(cid: str) -> FlowerClient:

    # Create model
    model = Net().to(DEVICE)

    # Load data (CIFAR-10)
    trainloaders, valloaders, testloader = load_datasets()

    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(cid, model, trainloader, valloader, testloader)

# Start Flower server for NUM_ROUNDS rounds of federated learning
if __name__ == "__main__":

    # Creates server's model
    model = Net().to(DEVICE)

    #fl.server.strategy.FedAvg
    strategy = fl.server.strategy.FedAvg(min_available_clients=MIN_AVAILABLE_CLIENTS,
        on_evaluate_config_fn=evaluate_config,
        eval_fn=get_eval_fn(model),
        #evaluate_metrics_aggregation_fn=agg
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