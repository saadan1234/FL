from typing import List, Tuple
from flwr.server import start_server
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
from visulalize import plot_metrics


# Initialize lists to store loss and accuracy values
rounds = []
loss = []
accuracy = []

# Define metric aggregation function to
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# Custom FedAvg strategy with a callback to store metrics
class CustomFedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def aggregate_evaluate(self, rnd: int, results, failures):
        # Call the aggregate_evaluate function from the superclass
        aggregated_result = super().aggregate_evaluate(rnd, results, failures)
        
        # Append the current round number
        rounds.append(rnd)
        
        # Extract the aggregated loss and accuracy from the result
        if aggregated_result is not None:
            loss_value = aggregated_result[0]
            metrics = aggregated_result[1]
            loss.append(loss_value)
            accuracy.append(metrics["accuracy"])
        
        return aggregated_result

# Define the server function
def server_fn(context):
    strategy = CustomFedAvg(evaluate_metrics_aggregation_fn=weighted_average)
    server_config = ServerConfig(num_rounds=20)
    return ServerAppComponents(
        strategy=strategy,
        server_config=server_config,
    )

# Flower ServerApp
app = ServerApp(server_fn=server_fn)


# Legacy mode
if __name__ == "__main__":
    # Start the Flower server
    start_server(
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=20),
        strategy=CustomFedAvg(evaluate_metrics_aggregation_fn=weighted_average),
    )

    # Plot the metrics after the server has completed training rounds
    plot_metrics(rounds, loss, accuracy)
