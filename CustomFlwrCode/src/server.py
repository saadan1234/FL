from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAvg
import numpy as np
from serverutils import detect_anomalies_zscore, fit_metrics_weighted_average, load_config, plot_training_metrics, weighted_average

# Global lists to track rounds, loss, and accuracy
rounds = []
loss = []
accuracy = []


class CustomFedAvg(FedAvg):
    def __init__(self, zscore_threshold: float = 2.5, momentum: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.zscore_threshold = zscore_threshold
        self.momentum = momentum
        self.previous_update = None  # To store previous update for momentum application
        self.previous_accuracy = None  # For smoothing global accuracy

    def aggregate_evaluate(self, rnd: int, results, failures):
        """Aggregate evaluation results and detect anomalies."""
        aggregated_result = super().aggregate_evaluate(rnd, results, failures)

        # Track round data
        rounds.append(rnd)
        if aggregated_result:
            loss_value, metrics = aggregated_result

            # Smooth accuracy using momentum
            current_accuracy = metrics["accuracy"]
            if self.previous_accuracy is not None:
                smoothed_accuracy = (
                    self.momentum * self.previous_accuracy
                    + (1 - self.momentum) * current_accuracy
                )
            else:
                smoothed_accuracy = current_accuracy

            self.previous_accuracy = smoothed_accuracy

            # Store smoothed metrics
            loss.append(loss_value)
            accuracy.append(smoothed_accuracy)

        print(f"Round {rnd} - Loss: {loss_value}, Smoothed Accuracy: {accuracy[-1]}")

        # Collect client updates for anomaly detection
        client_updates = [res.parameters for res in results if hasattr(res, "parameters")]
        if not client_updates:
            print("No client updates available for anomaly detection.")
            return aggregated_result

        updates_matrix = np.array([np.concatenate([p.flatten() for p in update]) for update in client_updates])

        if updates_matrix.size == 0:
            print("Empty updates matrix, skipping anomaly detection.")
            return aggregated_result

        # Detect anomalies using Z-score
        zscore_anomalies = detect_anomalies_zscore(updates_matrix, self.zscore_threshold)

        # Exclude detected anomalies from aggregation
        filtered_results = [res for i, res in enumerate(results) if i not in zscore_anomalies]

        print(f"Round {rnd}: Detected {len(zscore_anomalies)} anomalous clients, excluded from aggregation.")

        # Apply momentum and aggregate filtered results
        filtered_updates = np.array([np.concatenate([p.flatten() for p in res.parameters]) for res in filtered_results])
        aggregated_update = np.mean(filtered_updates, axis=0)

        if self.previous_update is not None:
            aggregated_update = self.momentum * self.previous_update + (1 - self.momentum) * aggregated_update

        self.previous_update = aggregated_update

        # Reconstruct update parameters
        aggregated_update_params = [
            np.reshape(aggregated_update[i:i + len(layer)], layer.shape)
            for i, layer in enumerate(self.model.get_weights())
        ]

        return super().aggregate_fit(rnd, filtered_results, failures, parameters=aggregated_update_params)

def start_federated_server(num_rounds: int, zscore_threshold: float, momentum: float, server_address: str):
    """Start the federated server with the custom strategy."""
    start_server(
        server_address=server_address,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=CustomFedAvg(
            evaluate_metrics_aggregation_fn=weighted_average,
            # fit_metrics_aggregation_fn=fit_metrics_weighted_average, 
            zscore_threshold=zscore_threshold,
            momentum=momentum
        ),
    )



def main():
    """Main function to set up and run the federated learning server."""
    # Start the federated server with the custom strategy
    config = load_config("config.yaml")
    server_config = config["server"]

    # Start the server with YAML config
    start_federated_server(
        num_rounds=server_config["num_rounds"],
        zscore_threshold=server_config["zscore_threshold"],
        momentum=server_config["momentum"],
        server_address=server_config["server_address"],
    )

    # Plot the metrics after training rounds are completed
    plot_training_metrics(rounds, loss, accuracy)

if __name__ == "__main__":
    main()
