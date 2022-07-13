
import math
import random
import flwr as fl
import numpy as np

from sklearn.cluster import KMeans

from logging import WARNING, INFO, DEBUG
from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

class HalfOfWeightsStrategy(fl.server.strategy.FedAvg):

    def __init__(self,min_available_clients): #fraction_fit,fraction_eval,min_fit_clients,min_eval_clients):

      super().__init__(min_available_clients=min_available_clients,
      #fraction_fit=fraction_fit,
      #fraction_eval=fraction_eval,
      #min_fit_clients=min_fit_clients,
      #min_eval_clients=min_eval_clients,
      )

    def aggregate_fit(self, rnd, results, failures):
      """Aggregate fit results using weighted average for half of the clients."""
        
      log(INFO, 'Using Half Of Weights Strategy')

      # Sample half of clients for aggregation
      half_clients = random.sample(results, int(len(results)/2))

      parameters_aggregated, metrics_aggregated = super().aggregate_fit(rnd, results, failures)
      log(DEBUG, 'All clients')
      log(DEBUG, parameters_aggregated)

      half_parameters_aggregated, half_metrics_aggregated = super().aggregate_fit(rnd, half_clients, failures)
      log(DEBUG, 'Half clients')
      log(DEBUG, half_parameters_aggregated)
      
      return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(self, rnd, results, failures):
      """Aggregate evaluation results."""

      # Sample half of clients for aggregation
      half_clients = random.sample(results, int(len(results)/2))

      log(DEBUG, len(half_clients))
      return super().aggregate_evaluate(rnd, half_clients, failures)


class ClusterStrategy(fl.server.strategy.FedAvg):

    def __init__(self,min_available_clients): #fraction_fit,fraction_eval,min_fit_clients,min_eval_clients):

      super().__init__(min_available_clients=min_available_clients,
      #fraction_fit=fraction_fit,
      #fraction_eval=fraction_eval,
      #min_fit_clients=min_fit_clients,
      #min_eval_clients=min_eval_clients,
      )

    def aggregate_fit(self, rnd, results, failures):
      """Aggregate fit results using weighted average for half of the clients."""
        
      log(INFO, 'Using Half Of Weights Strategy')
      log(DEBUG, len(results))

      # Sample half of clients for aggregation
      half_clients = random.sample(results, int(len(results)/2))

      log(DEBUG, len(half_clients))
      return super().aggregate_fit(rnd, half_clients, failures)

    def aggregate_evaluate(self, rnd, results, failures):
      """Aggregate evaluation results."""

      # Sample half of clients for aggregation
      half_clients = random.sample(results, int(len(results)/2))

      log(DEBUG, len(half_clients))
      return super().aggregate_evaluate(rnd, half_clients, failures)



