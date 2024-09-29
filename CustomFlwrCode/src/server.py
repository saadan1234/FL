from typing import List, Tuple, Dict
from flwr.server import start_server
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
from visulalize import plot_metrics

# Initialize lists to store loss and accuracy values
rounds = []
loss = []
accuracy = []

# Track client performance
client_performance: Dict[int, List[float]] = {}

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
        
        # Track individual client performance
        previous_accuracy = None
        for client_id, (num_examples, client_metrics) in enumerate(results):
            current_accuracy = client_metrics.metrics["accuracy"]
            if previous_accuracy is None or current_accuracy >= previous_accuracy:
                if client_id not in client_performance:
                    client_performance[client_id] = []
                client_performance[client_id].append(current_accuracy)
                previous_accuracy = current_accuracy
        
        # Identify and separate clients with degrading performance
        self.separate_degrading_clients()
        
        return aggregated_result
    
    def separate_degrading_clients(self):
        degrading_clients = []
        for client_id, performances in client_performance.items():
            if len(performances) >= 3 and all(x > y for x, y in zip(performances[-3:], performances[-2:])):
                degrading_clients.append(client_id)
        
        if degrading_clients:
            print(f"Separating clients with degrading performance: {degrading_clients}")
            for client_id in degrading_clients:
                del client_performance[client_id]

# Legacy mode
if __name__ == "__main__":
    # Start the Flower server
    start_server(
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=16),
        strategy=CustomFedAvg(evaluate_metrics_aggregation_fn=weighted_average),
    )

    # Plot the metrics after the server has completed training rounds
    plot_metrics(rounds, loss, accuracy)