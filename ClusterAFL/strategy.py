
import math
import utils
import random
import flwr as fl
import numpy as np

from collections.abc import Iterable
from itertools import chain
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

from flwr.server.strategy.aggregate import weighted_loss_avg

class HalfOfWeightsStrategy(fl.server.strategy.FedAvg):

    def __init__(self, min_available_clients, on_evaluate_config_fn, eval_fn, fraction_fit, fraction_eval, min_fit_clients, min_eval_clients):

      super().__init__(min_available_clients=min_available_clients,
        on_evaluate_config_fn=on_evaluate_config_fn, eval_fn=eval_fn,
        fraction_fit=fraction_fit, fraction_eval=fraction_eval,
        min_fit_clients=min_fit_clients, min_eval_clients=min_eval_clients)

    def aggregate_fit(self, rnd, results, failures):
      """Aggregate fit results using weighted average for half of the clients."""

      # Sample half of clients for aggregation
      # half_clients = random.sample(results, len(results)//2)
      half_clients = results[:len(results)//2]

      log(DEBUG, 'Using half of clients')
      log(DEBUG, len(half_clients))

      return super().aggregate_fit(rnd, half_clients, failures)

    # def aggregate_evaluate(self, rnd, results, failures):
    #   """Aggregate evaluation results."""

    #   # Sample half of clients for aggregation
    #   # half_clients = random.sample(results, int(len(results)/2))
    #   half_clients = results[:len(results)//2]
    #   return super().aggregate_evaluate(rnd, half_clients, failures)

    def aggregate_evaluate(self, rnd, results, failures):
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        half_clients = results[:len(results)//2]
        log(DEBUG, 'Using half of clients')
        log(DEBUG, len(half_clients))

        loss_aggregated = weighted_loss_avg(
            [
                (
                    evaluate_res.num_examples,
                    evaluate_res.loss,
                )
                for _, evaluate_res in half_clients
            ]
        )
        accuracy_aggregated = weighted_loss_avg(
            [
                (
                    evaluate_res.num_examples,
                    evaluate_res.metrics['accuracy'],
                )
                for _, evaluate_res in half_clients
            ]
        )
        return loss_aggregated, {'accuracy': accuracy_aggregated}


class ClusterStrategy(fl.server.strategy.FedAvg):

    def __init__(self, min_available_clients, on_evaluate_config_fn, eval_fn, fraction_fit, fraction_eval, min_fit_clients, min_eval_clients, n_clusters, random_state=0):

      super().__init__(min_available_clients=min_available_clients,
        on_evaluate_config_fn=on_evaluate_config_fn, eval_fn=eval_fn,
        fraction_fit=fraction_fit, fraction_eval=fraction_eval,
        min_fit_clients=min_fit_clients, min_eval_clients=min_eval_clients)

      self.n_clusters = n_clusters
      self.random_state = random_state

    def aggregate_fit(self, rnd, results, failures):
      """Aggregate fit results using weighted average for half of the clients."""

      # Convert results and flat weights
      X = []
      for _, fit_res in results:
        X.append(utils.concatenate_arrays_recursive(parameters_to_weights(fit_res.parameters), result=[]))

      kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(X)

      clusters = []
      for i in range(self.n_clusters):
        clusters.append([])

      for i in range(len(kmeans.labels_)):
        clusters[kmeans.labels_[i]].append(results[i])
        
      return super().aggregate_fit(rnd, results, failures)

    def aggregate_evaluate(self, rnd, results, failures):
      """Aggregate evaluation results."""

      return super().aggregate_evaluate(rnd, results, failures)



