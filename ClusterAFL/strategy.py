
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

    def __init__(self, min_available_clients): #fraction_fit,fraction_eval,min_fit_clients,min_eval_clients):

      super().__init__(min_available_clients=min_available_clients)

    def aggregate_fit(self, rnd, results, failures):
      """Aggregate fit results using weighted average for half of the clients."""

      # Sample half of clients for aggregation
      # half_clients = random.sample(results, len(results)//2)
      half_clients = results[:len(results)//2]

      return super().aggregate_fit(rnd, half_clients, failures)

    def aggregate_evaluate(self, rnd, results, failures):
      """Aggregate evaluation results."""

      # Sample half of clients for aggregation
      # half_clients = random.sample(results, int(len(results)/2))
      half_clients = results[:len(results)//2]

      return super().aggregate_evaluate(rnd, half_clients, failures)

    # def aggregate_evaluate(
    #     self,
    #     rnd: int,
    #     results: List[Tuple[ClientProxy, EvaluateRes]],
    #     failures: List[BaseException],
    # ) -> Optional[float]:
    #     """Aggregate evaluation losses using weighted average."""
    #     if not results:
    #         return None

    #     # Weigh accuracy of each client by number of examples used
    #     accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
    #     examples = [r.num_examples for _, r in results]

    #     # Aggregate and print custom metric
    #     accuracy_aggregated = sum(accuracies) / sum(examples)
    #     print(f"Round {rnd} accuracy aggregated from client results: {accuracy_aggregated}")

    #     # Call aggregate_evaluate from base class (FedAvg)
    #     return super().aggregate_evaluate(rnd, results, failures)


class ClusterStrategy(fl.server.strategy.FedAvg):

    def __init__(self, min_available_clients):

      super().__init__(min_available_clients=min_available_clients)

    def aggregate_fit(self, rnd, results, failures):
      """Aggregate fit results using weighted average for half of the clients."""

      # Convert results
      X = [
          (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
          for _, fit_res in results
      ]

      log(DEBUG, 'K-Means')
      log(DEBUG, len(X))

      kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
      log(DEBUG, 'K-Means')
      log(DEBUG, kmeans.labels_)
        
      return super().aggregate_fit(rnd, results, failures)

    def aggregate_evaluate(self, rnd, results, failures):
      """Aggregate evaluation results."""

      return super().aggregate_evaluate(rnd, results, failures)



