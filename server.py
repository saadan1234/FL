from typing import List, Tuple
from flwr.server import start_server
from flwr.server import ServerApp,ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define the server function
def server_fn(context):
    strategy = FedAvg(evaluate_metrics_aggregation_fn=weighted_average)
    server_config = ServerConfig(num_rounds=10)
    return ServerAppComponents(
        strategy=strategy,
        server_config=server_config,
    )

# Flower ServerApp
app = ServerApp(server_fn=server_fn)

# Legacy mode
if __name__ == "__main__":


    start_server(
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=20),
        strategy=FedAvg(evaluate_metrics_aggregation_fn=weighted_average),
    )