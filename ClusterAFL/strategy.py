
import flwr as fl
import numpy as np

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

#from .aggregate import aggregate, weighted_loss_avg
#from .strategy import Strategy

class HalfOfWeightsStrategy(fl.server.strategy.FedAvg):

    def __init__(self,min_available_clients,eval_fn,on_fit_config_fn):

        super().__init__(min_available_clients=min_available_clients, 
        eval_fn=eval_fn,
        on_fit_config_fn=on_fit_config_fn)

    def aggregate_fit(self, rnd, results, failures):
        """Aggregate fit results using weighted average."""
        
        log(INFO, 'Using Half Of Weights Strategy')

        if not results:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        parameters_aggregated = weights_to_parameters(super().aggregate(weights_results))