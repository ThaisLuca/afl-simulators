
import utils
import client
import server
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

server = Server(model,model_weights_type)

model_weights_type = server.server_init.type_signature.result

federated_server_type = tff.FederatedType(model_weights_type, tff.SERVER)
federated_dataset_type = tff.FederatedType(tf_dataset_type, tff.CLIENTS)

# Implementa uma rodada de comunicação
@tff.federated_computation(federated_server_type, federated_dataset_type)
def next_fn(server_weights, federated_dataset):
  # Broadcast the server weights to the clients.
  server_weights_at_client = tff.federated_broadcast(server_weights)

  # Each client computes their updated weights.
  client_weights = tff.federated_map(client_update_fn, (federated_dataset, server_weights_at_client))

  # The server averages these updates.
  mean_client_weights = tff.federated_mean(client_weights)

  # The server updates its model.
  server_weights = tff.federated_map(server_update_fn, mean_client_weights)

  return server_weights

federated_algorithm = tff.templates.IterativeProcess(initialize_fn=server.initialize_fn, next_fn=next_fn)