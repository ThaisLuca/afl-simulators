
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
    NDArrays,
    parameters_to_ndarrays,
    ndarrays_to_parameters
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from flwr.server.strategy.aggregate import weighted_loss_avg, aggregate

class ClusterStrategy(fl.server.strategy.FedAvg):

    def __init__(self, min_available_clients, 
        on_evaluate_config_fn, evaluate_fn, fraction_fit, 
        fraction_evaluate, min_fit_clients, min_evaluate_clients, 
        n_clusters, random_state=0):

      super().__init__(min_available_clients=min_available_clients,
        on_evaluate_config_fn=on_evaluate_config_fn, evaluate_fn=evaluate_fn,
        fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients)

      self.n_clusters = n_clusters
      self.random_state = random_state
      self.kmeans = None

    def aggregate(results):
      """Compute weighted average."""
      # Calculate the total number of examples used during training
      num_examples_total = sum([num_examples for _, num_examples in results])

      # Create a list of weights, each multiplied by the related number of examples
      weighted_weights = [
          [layer * num_examples for layer in weights] for weights, num_examples in results
      ]

      # Compute average weights of each layer
      weights_prime: NDArrays = [
          reduce(np.add, layer_updates) / num_examples_total
          for layer_updates in zip(*weighted_weights)
      ]
      return weights_prime      

    def aggregate_fit(self, server_round, results, failures):
      """Aggregate fit results using weighted average for half of the clients."""

      if not results:
            return None, {}
      # Do not aggregate if there are failures and failures are not accepted
      if not self.accept_failures and failures:
          return None, {}

      # Convert results and flat weights
      X = []
      # ver quem é o primeiro parâmetro
      # uma linha pra cada nó pra todos os nós controlando a ordem que eles entram na matriz (ordenados pelo ID)
      # matriz tem linha toda NaN se dropout
      for _, fit_res in results:
        X.append(utils.concatenate_arrays_recursive(parameters_to_ndarrays(fit_res.parameters), result=[]))

      self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(X)

      clusters = []
      for i in range(self.n_clusters):
        clusters.append([])

      for i in range(len(self.kmeans.labels_)):
        clusters[self.kmeans.labels_[i]].append(results[i])

    # cluster -> lista de listas de parâmetros (dicionário)

      clustering_results = []
      for i in range(self.n_clusters):
        parameters_aggregated = ndarrays_to_parameters(aggregate(clusters[i]))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in clusters[i]]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        clustering_results.append((parameters_aggregated, metrics_aggregated))
      return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation losses using weighted average."""
        
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        clusters = []
        for i in range(self.n_clusters):
          clusters.append([])

        for i in range(len(self.kmeans.labels_)):
          clusters[self.kmeans.labels_[i]].append(results[i])

        loss_aggregated = weighted_loss_avg(
            [
                (
                    evaluate_res.num_examples,
                    evaluate_res.loss,
                )
                for _, evaluate_res in results
            ]
        )
        accuracy_aggregated = weighted_loss_avg(
            [
                (
                    evaluate_res.num_examples,
                    evaluate_res.metrics['accuracy'],
                )
                for _, evaluate_res in results
            ]
        )
        return loss_aggregated, {'accuracy': accuracy_aggregated}

    def configure_fit(self, server_round, parameters, client_manager: ClientManager):
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]


