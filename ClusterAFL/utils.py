
import torch
import pickle

from collections import OrderedDict
from typing import List

from collections.abc import Iterable
import numpy as np

DEVICE = torch.device("cpu")

NUM_CLIENTS = 10
NUM_ROUNDS = 3
BATCH_SIZE = 32
MIN_AVAILABLE_CLIENTS = int(NUM_CLIENTS * 0.80)    # Wait until at least 75 clients are available
FRACTION_FIT = 0                                   # Sample 100% of available clients for training
MIN_FIT_CLIENTS = 2                                # Never sample less than 50 clients for training
FRACTION_EVAL = 1                                  # Sample 100% of available clients for evaluation
MIN_EVAL_CLIENTS = 2                               # Never sample less than 50 clients for evaluation

N_CLUSTERS = 1                                     # Number of clusters to split clients
EPOCHS = 1                                         # Number of local rounds
SEED = 42

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
      epoch_loss /= len(trainloader)
      epoch_acc = correct / total
      if verbose:
          print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
  return epoch_loss, epoch_acc

  # if valloader is not None:
  #   loss, accuracy = test(net, valloader)
  #   if verbose:
  #     print(f"Local round val loss {loss}, accuracy {accuracy}")

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
  loss /= len(testloader)
  accuracy = correct / total
  return loss, accuracy

def set_parameters(model: torch.nn.ModuleList, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def concatenate_arrays_recursive(element, result=[]):
  if(isinstance(element, Iterable)):
    for el in element:
      result = concatenate_arrays_recursive(el, result)
    return result
  elif(not isinstance(element, Iterable)):
    result.append(element)
    return result
  else:
    return result

def save_pickle_file(weights):
  with open('weights.pkl', 'wb') as file:
    pickle.dump(weights, file)

def load_pickle_file():
    with open('weights.pkl', 'rb') as file:
        return pickle.load(file)