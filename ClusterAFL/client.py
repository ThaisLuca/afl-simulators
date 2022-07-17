
import torch
import warnings
import flwr as fl
import numpy as np
import tensorflow as tf

from collections import OrderedDict
from collections import OrderedDict
from typing import List

DEVICE = torch.device("cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, ID, model, trainloader, valloader):
        self.ID = ID
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

    def train(self, epochs: int, verbose=False):
      """Train the network on the training set."""
      criterion = torch.nn.CrossEntropyLoss()
      optimizer = torch.optim.Adam(self.model.parameters())
      self.model.train()
      for epoch in range(epochs):
          correct, total, epoch_loss = 0, 0, 0.0
          for images, labels in self.trainloader:
              images, labels = images.to(DEVICE), labels.to(DEVICE)
              optimizer.zero_grad()
              outputs = self.model(images)
              loss = criterion(self.model(images), labels)
              loss.backward()
              optimizer.step()
              # Metrics
              epoch_loss += loss
              total += labels.size(0)
              correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
          epoch_loss /= len(self.testloader.dataset)
          epoch_acc = correct / total
          if verbose:
              print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

    def test(self):
      """Evaluate the network on the entire test set."""
      criterion = torch.nn.CrossEntropyLoss()
      correct, total, loss = 0, 0, 0.0
      self.net.eval()
      with torch.no_grad():
          for images, labels in self.valloader:
              images, labels = images.to(DEVICE), labels.to(DEVICE)
              outputs = self.model(images)
              loss += criterion(outputs, labels).item()
              _, predicted = torch.max(outputs.data, 1)
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
      loss /= len(self.testloader.dataset)
      accuracy = correct / total
      return loss, accuracy

    def get_parameters(self) -> List[np.ndarray]:
      return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train(epochs=1, verbose=True)
        return self.get_parameters(), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.test()
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}