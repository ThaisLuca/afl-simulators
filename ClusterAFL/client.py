
import torch
import warnings
import flwr as fl
import numpy as np

from collections import OrderedDict
from typing import List

from flwr.common.logger import log
from logging import WARNING, INFO, DEBUG
from utils import train, test, EPOCHS

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, ID, model, trainloader, valloader):
        self.ID = ID
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
      return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = train(self.model, self.trainloader, self.valloader, epochs=EPOCHS, verbose=True)
        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}