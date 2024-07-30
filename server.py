import pickle
from typing import List, Tuple

from flwr.server import ServerApp,ServerAppComponents, ServerConfig, start_server
from flwr.server.strategy import FedAvg
from flwr.common import Metrics

from client import get_model

with open('data.pkl', 'rb') as f:
    X_train, Y_train, X_test, Y_test = pickle.load(f)

# Example function to simulate model evaluation
def evaluate_model(model, X_test, Y_test):
    # Simulating evaluation; in practice, use model.evaluate(x_test, y_test)
    return model.evaluate(X_test, Y_test, verbose=0)

# Define metric aggregation function for both MSE and generic loss
def weighted_average(models, X_test, Y_test, weights=None):
    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    total_weighted_metric = 0.0
    total_weight = 0.0

    for model, weight in zip(models, weights):
        evaluation = evaluate_model(model, X_test, Y_test)
        if len(evaluation) != 2:
            raise ValueError("Expected evaluation to return 2 values: loss and metric")

        loss, metric = evaluation
        total_weighted_metric += weight * metric
        total_weight += weight

    return total_weighted_metric / total_weight

# Define the server function
def server_fn(context):
    strategy = FedAvg(evaluate_metrics_aggregation_fn=weighted_average(get_model(), X_test, Y_test))
    server_config = ServerConfig(num_rounds=3)
    return ServerAppComponents(
        strategy=strategy,
        server_config=server_config,
    )

# Flower ServerApp
# app = ServerApp(server_fn=server_fn)

# Legacy mode
if __name__ == "__main__":
    start_server(
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=3),
        strategy=FedAvg(evaluate_metrics_aggregation_fn=weighted_average(get_model(), X_test, Y_test)),
    )