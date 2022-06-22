
import utils
import random
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

model_weights_type = utils.model.type_signature.result

class Server:

  def __init__():
    self.__ID = random.randint(100,200)

  @tff.tf_computation(model_weights_type)
  def server_update_fn(self, mean_client_weights):
    return server_update(model, mean_client_weights)

  @tff.tf_computation
  def server_init(self):
    return model.trainable_variables

  # Inicializa o servidor
  @tff.federated_computation
  def initialize_fn(self):
    return tff.federated_value(self.server_init(), tff.SERVER)

  @tf.function
  def server_update(self, mean_client_weights):
    """Updates the server model weights as the average of the client model weights."""
    
    model_weights = model.trainable_variables

    # Assign the mean client weights to the server model (FedAvg)
    tf.nest.map_structure(lambda x, y: x.assign(y), model_weights, mean_client_weights)
    return model_weights