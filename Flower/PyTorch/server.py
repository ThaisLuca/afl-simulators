
import flwr as fl
import os

# Primeiro inicia o servidor
fl.server.start_server(server_address="[::]:4466",config={"num_rounds": 3})