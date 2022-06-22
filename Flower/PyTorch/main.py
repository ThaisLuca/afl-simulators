
import flwr as fl
import os

os.system ("echo 'Hello!'")

# Primeiro inicia o servidor
fl.server.start_server(server_address="[::]:4466",config={"num_rounds": 3})

# Cada cliente precisa ser iniciado em um terminal diferente. No caso de 1 servidor e 2 clientes, 
# são necessários três terminais diferentes. A versão usando o PyTorch parece não permitir modificações
# nas mensagens e em criar mais de um servidor, por exemplo.

# Cliente 1
fl.client.start_numpy_client("127.0.0.1:4466", client=CifarClient())

# Cliente 2
fl.client.start_numpy_client("127.0.0.1:4466", client=CifarClient())