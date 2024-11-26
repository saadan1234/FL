from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAvg
import numpy as np
from serverutils import detect_anomalies_zscore, load_config, plot_training_metrics, weighted_average
from crypto.rsa_crypto import RsaCryptoAPI
from flwr.common import (
    Parameters,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    Code)
# Global lists to track rounds, loss, and accuracy
rounds = []
loss = []
accuracy = []

class CustomFedAvg(FedAvg):
    def __init__(self, zscore_threshold: float = 2.5, momentum: float = 0.9, aes_key: bytes = None, **kwargs):
        super().__init__(**kwargs)
        self.zscore_threshold = zscore_threshold
        self.momentum = momentum
        self.previous_update = None  # To store previous update for momentum application
        self.previous_accuracy = None  # For smoothing global accuracy
        self.aes_key = aes_key  # AES key for decryption and encryption

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

        return aggregated_result

    def aggregate_fit(self, rnd: int, results, failures):
        """Aggregate fit results and detect anomalies."""
        # Collect client updates for anomaly detection
        client_updates = [res.parameters for res in results if hasattr(res, "parameters")]

        if not client_updates:
            print("No client updates available for anomaly detection.")
            return super().aggregate_fit(rnd, results, failures)

        # Decrypt client updates
        decrypted_updates = []
        for update in client_updates:
            decrypted_update = [np.frombuffer(RsaCryptoAPI.decrypt_obj(self.aes_key, param), dtype=np.float32).reshape(param.shape) for param in update.tensors]
            decrypted_updates.append(decrypted_update)

        updates_matrix = np.array([np.concatenate([p.flatten() for p in update]) for update in decrypted_updates])

        if updates_matrix.size == 0:
            print("Empty updates matrix, skipping anomaly detection.")
            return super().aggregate_fit(rnd, results, failures)

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

        # Encrypt the aggregated update parameters
        enc_aggregated_update_params = [RsaCryptoAPI.encrypt_numpy_array(self.aes_key, param) for param in aggregated_update_params]

        return super().aggregate_fit(rnd, filtered_results, failures, parameters=Parameters(tensors=enc_aggregated_update_params, tensor_type=""))

def start_federated_server(num_rounds: int, zscore_threshold: float, momentum: float, server_address: str, aes_key: bytes):
    """Start the federated server with the custom strategy."""
    start_server(
        server_address=server_address,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=CustomFedAvg(
            evaluate_metrics_aggregation_fn=weighted_average,
            zscore_threshold=zscore_threshold,
            momentum=momentum,
            aes_key=aes_key
        ),
    )

def main():
    """Main function to set up and run the federated learning server."""
    # Start the federated server with the custom strategy
    config = load_config("config.yaml")
    server_config = config["server"]
    aes_key = RsaCryptoAPI.load_key('aes_key.bin')  # Load the AES key

    # Start the server with YAML config
    start_federated_server(
        num_rounds=server_config["num_rounds"],
        zscore_threshold=server_config["zscore_threshold"],
        momentum=server_config["momentum"],
        server_address=server_config["server_address"],
        aes_key=aes_key
    )

    # Plot the metrics after training rounds are completed
    plot_training_metrics(rounds, loss, accuracy)

if __name__ == "__main__":
    main()