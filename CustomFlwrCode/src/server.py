from typing import List, Tuple, Dict
from flwr.server import start_server
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
from visualize import plot_metrics
import numpy as np
from scipy.stats import zscore
import tensorflow as tf

# Initialize lists to store loss and accuracy values for visualization
rounds = []
loss = []
accuracy = []

# Track client performance
client_performance: Dict[int, List[float]] = {}

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# Autoencoder model for anomaly detection
def create_autoencoder(input_dim):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(input_dim, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# Example training on historical data
historical_updates = np.random.rand(1000, 100)  # Example of historical client updates
autoencoder = create_autoencoder(historical_updates.shape[1])
autoencoder.fit(historical_updates, historical_updates, epochs=10, batch_size=32)

# Detect anomalies based on Z-score
def detect_anomalies_zscore(updates, threshold=2.5):
    mean_update = np.mean(updates, axis=0)
    z_scores = np.array([zscore(update) for update in updates])
    anomalies = [i for i, z in enumerate(z_scores) if np.any(np.abs(z) > threshold)]
    return anomalies

# Detect anomalies based on Autoencoder reconstruction error
def detect_anomalies_autoencoder(model, updates, threshold=0.05):
    if updates.size == 0:
        # Return an empty list if there are no updates to process
        print("No updates to process for autoencoder anomaly detection.")
        return []
    
    reconstructions = model.predict(updates)
    reconstruction_errors = np.mean(np.square(updates - reconstructions), axis=1)
    anomalies = [i for i, error in enumerate(reconstruction_errors) if error > threshold]
    return anomalies


class CustomFedAvg(FedAvg):
    def __init__(self, zscore_threshold=2.5, reconstruction_threshold=0.05, **kwargs):
        super().__init__(**kwargs)
        self.zscore_threshold = zscore_threshold
        self.reconstruction_threshold = reconstruction_threshold
        self.autoencoder = autoencoder
    
    def aggregate_evaluate(self, rnd: int, results, failures):
        # Call the aggregate_evaluate function from the superclass
        aggregated_result = super().aggregate_evaluate(rnd, results, failures)
        
        # Append the current round number for tracking
        rounds.append(rnd)
        
        # Extract aggregated loss and accuracy from the result
        if aggregated_result is not None:
            loss_value = aggregated_result[0]
            metrics = aggregated_result[1]
            loss.append(loss_value)
            accuracy.append(metrics["accuracy"])

        # Print statements for debugging
        print(f"Round {rnd} - Loss: {loss_value}, Accuracy: {metrics['accuracy']}")
        
        # Collect client updates for anomaly detection
        client_updates = [res.parameters for res in results if hasattr(res, "parameters")]
        if not client_updates:
            print("No client updates available for anomaly detection.")
            return aggregated_result  # Skip anomaly detection if no updates

        updates_matrix = np.array([np.concatenate([p.flatten() for p in update]) for update in client_updates])
        
        # Ensure updates_matrix has valid data before detecting anomalies
        if updates_matrix.size == 0:
            print("Empty updates matrix, skipping anomaly detection.")
            return aggregated_result
        
        # Detect anomalies with Z-score
        zscore_anomalies = detect_anomalies_zscore(updates_matrix, self.zscore_threshold)
        
        # Detect anomalies with Autoencoder
        autoencoder_anomalies = detect_anomalies_autoencoder(self.autoencoder, updates_matrix, self.reconstruction_threshold)
        
        # Combine detected anomalies and isolate them
        anomalies = set(zscore_anomalies + autoencoder_anomalies)
        filtered_results = [res for i, res in enumerate(results) if i not in anomalies]
        
        print(f"Round {rnd}: Detected {len(anomalies)} anomalous clients, excluded from aggregation.")
        
        # Continue with aggregation on filtered results
        return super().aggregate_fit(rnd, filtered_results, failures)

# Visualization should be called after server rounds complete
if __name__ == "__main__":
    # Start the Flower server with the custom anomaly detection strategy
    start_server(
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=5),
        strategy=CustomFedAvg(
            evaluate_metrics_aggregation_fn=weighted_average,
            zscore_threshold=2.5,
            reconstruction_threshold=0.05
        ),
    )

    # Verify rounds, loss, and accuracy contain data before plotting
    print("Rounds:", rounds)
    print("Loss:", loss)
    print("Accuracy:", accuracy)

    # Plot the metrics after the server has completed training rounds
    plot_metrics(rounds, loss, accuracy)
