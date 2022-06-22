
import random
import numpy as np

import utils
import tensorflow as tf
import tensorflow_federated as tff

tf_dataset_type = tff.SequenceType(utils.model.input_spec)
model_weights_type = utils.model.type_signature.result

class Client:

  def __init__():
    self.__ID = random.randint(100,200)

  @tff.tf_computation(tf_dataset_type, model_weights_type)
  def client_update_fn(self):
    client_optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    return client_update(model, tf_dataset, server_weights, client_optimizer)

  @tf.function
  def client_update(self, model, dataset, server_weights, client_optimizer):
    """Performs training (using the server model weights) on the client's dataset."""
    
    # Initialize the client model with the current server weights.
    client_weights = model.trainable_variables
    
    # Assign the server weights to the client model.
    tf.nest.map_structure(lambda x, y: x.assign(y), client_weights, server_weights)

    # Use the client_optimizer to update the local model.
    for batch in dataset:
      with tf.GradientTape() as tape:
        
        # Compute a forward pass on the batch of data
        outputs = model.forward_pass(batch)

      # Compute the corresponding gradient
      grads = tape.gradient(outputs.loss, client_weights)
      grads_and_vars = zip(grads, client_weights)

      # Apply the gradient using a client optimizer.
      client_optimizer.apply_gradients(grads_and_vars)

    return client_weights