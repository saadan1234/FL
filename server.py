from typing import List, Tuple

from flwr.server import ServerApp,ServerAppComponents, ServerConfig, start_server
from flwr.server.strategy import FedAvg
from flwr.common import Metrics

# Define metric aggregation function for both MSE and generic loss
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Determine which metric to use
    metric_name = None
    if "mse" in metrics[0][1]:
        metric_name = "mse"
    elif "loss" in metrics[0][1]:
        metric_name = "loss"

    if metric_name is None:
        raise ValueError("Metrics must contain either 'mse' or 'loss' key.")

    # Multiply the metric of each client by the number of examples used
    weighted_metrics = [num_examples * m[metric_name] for num_examples, m in metrics if metric_name in m]
    examples = [num_examples for num_examples, m in metrics if metric_name in m]

    # Check if we have valid examples
    if not examples:
        raise ValueError(f"No valid metrics found for '{metric_name}'.")

    # Aggregate and return custom metric (weighted average)
    return {metric_name: sum(weighted_metrics) / sum(examples)}


# Define the server function
def server_fn(context):
    strategy = FedAvg(evaluate_metrics_aggregation_fn=weighted_average)
    server_config = ServerConfig(num_rounds=3)
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
        config=ServerConfig(num_rounds=3),
        strategy=FedAvg(evaluate_metrics_aggregation_fn=weighted_average),
    )