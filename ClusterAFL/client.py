
import torch
import warnings
import flwr as fl
import numpy as np

from collections import OrderedDict
from typing import List

from utils import train, test

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, ID, model, trainloader, valloader, testloader):
        self.ID = ID
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader

    def get_parameters(self) -> List[np.ndarray]:
      return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.trainloader, self.valloader, epochs=1, verbose=True)
        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader)
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}