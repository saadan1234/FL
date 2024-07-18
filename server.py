import flwr as fl
from flwr.server import ServerConfig
from flwr.server.strategy import FedMedian

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=ServerConfig(num_rounds=3),
    strategy=FedMedian(),
    )