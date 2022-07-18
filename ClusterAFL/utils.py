
import torch

from collections import OrderedDict
from typing import List

import numpy as np

DEVICE = torch.device("cpu")

NUM_CLIENTS = 100
NUM_ROUNDS = 3
BATCH_SIZE = 32
MIN_AVAILABLE_CLIENTS = 2 # int(NUM_CLIENTS * 0.75)# Wait until at least 75 clients are available
#FRACTION_FIT = 0.1                                # Sample 10% of available clients for training
#MIN_FIT_CLIENTS = 10                              # Never sample less than 10 clients for training
#FRACTION_EVAL = 0.05                              # Sample 5% of available clients for evaluation
#MIN_EVAL_CLIENTS = 5                              # Never sample less than 5 clients for evaluation

def train(net, trainloader, valloader, epochs: int, verbose=False):
  """Train the network on the training set."""
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(net.parameters())
  net.train()
  for epoch in range(epochs):
      correct, total, epoch_loss = 0, 0, 0.0
      for images, labels in trainloader:
          images, labels = images.to(DEVICE), labels.to(DEVICE)
          optimizer.zero_grad()
          outputs = net(images)
          loss = criterion(net(images), labels)
          loss.backward()
          optimizer.step()
          # Metrics
          epoch_loss += loss
          total += labels.size(0)
          correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
      epoch_loss /= len(valloader.dataset)
      epoch_acc = correct / total
      if verbose:
          print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
  """Evaluate the network on the entire test set."""
  criterion = torch.nn.CrossEntropyLoss()
  correct, total, loss = 0, 0, 0.0
  net.eval()
  with torch.no_grad():
      for images, labels in testloader:
          images, labels = images.to(DEVICE), labels.to(DEVICE)
          outputs = net(images)
          loss += criterion(outputs, labels).item()
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  loss /= len(testloader.dataset)
  accuracy = correct / total
  return loss, accuracy

def set_parameters(model, parameters: List[np.ndarray]):
  params_dict = zip(model.state_dict().keys(), parameters)
  state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
  model.load_state_dict(state_dict, strict=True)