from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from network import Net

import flwr as fl
import random 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CifarClient_Server(fl.client.NumPyClient):

  def __init__(self):
    # Load model and data
    print('Client Server %s starting' % str(random.randint(100,200)))
    self.net = Net().to(DEVICE)
    self.trainloader, self.testloader, self.num_examples = self.load_data()

  def get_parameters(self):
    return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

  def set_parameters(self, parameters):
    params_dict = zip(self.net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    self.net.load_state_dict(state_dict, strict=True)

  def fit(self, parameters, config):
    self.set_parameters(parameters)
    self.train(self.net, self.trainloader, epochs=1)
    return self.get_parameters(), self.num_examples["trainset"], {}

  def evaluate(self, parameters, config):
    self.set_parameters(parameters)
    loss, accuracy = self.test(self.net, self.testloader)
    return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

  def load_data(self):
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10(".", train=True, download=True, transform=transform)
    testset = CIFAR10(".", train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    num_examples = {"trainset" : len(trainset), "testset" : len(testset)}
    return trainloader, testloader, num_examples

  def train(self, net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
      for images, labels in trainloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(net(images), labels)
        loss.backward()
        optimizer.step()

  def test(self, net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
      for data in testloader:
        images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
        outputs = net(images)
        loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy

fl.server.start_server(server_address="[::]:5566",config={"num_rounds": 3})

fl.client.start_numpy_client("127.0.0.1:5566", client=CifarClient_Server())

fl.client.start_numpy_client("127.0.0.1:5566", client=CifarClient_Server())