from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAvg
import numpy as np
from serverutils import detect_anomalies_zscore, load_config, plot_training_metrics, weighted_average
from crypto.rsa_crypto import RsaCryptoAPI
from transformers import AutoTokenizer
from model import build_model
from clientutils import load_data, load_dataset_hf, prepare_data, preprocess_and_split, save_data
from flwr.common import (
    Parameters,
    NDArrays,
)

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
        self.model = None  # Placeholder for the model
        self.original_weights = None

    def _decrypt_params(self, parameters: Parameters) -> NDArrays:
        params = zip(parameters.tensors,
                    [w.shape for w in self.model.get_weights()],
                    [w.dtype for w in self.model.get_weights()])

        decrypted_params = []
        for param, shape, dtype in params:
            encrypted_size = len(param)
            expected_size = np.prod(shape) * np.dtype(dtype).itemsize
            print(f"Encrypted param size: {encrypted_size}, expected size: {expected_size}, shape: {shape}, dtype: {dtype}")

            decrypted_param = RsaCryptoAPI.decrypt_obj(self.aes_key, param)
            print(f"Decrypted param size: {len(decrypted_param)}, expected shape: {shape}, dtype: {dtype}")
            
            decrypted_param = np.frombuffer(buffer=decrypted_param, dtype=dtype)
            print(f"Decrypted param reshaped size: {decrypted_param.size}")
            
            decrypted_param = decrypted_param.reshape(shape)
            decrypted_params.append(decrypted_param)

        return decrypted_params
    def load_and_prepare_data(self, dataset_name, dataset_type='traditional', input_column=None, output_column=None):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") if dataset_type == 'text' else None

        # Load and prepare the dataset
        dataset = load_dataset_hf(dataset_name, input_column=input_column, output_column=output_column, dataset_type=dataset_type)
        dataset = prepare_data(dataset, tokenizer, input_col=input_column, output_col=output_column, dataset_type=dataset_type)
        X_train, X_test, Y_train, Y_test = preprocess_and_split(dataset['train'], tokenizer, dataset_type, 
                                                                input_col=input_column, output_col=output_column)

        # Save and reload data for simplicity (optional)
        save_data('data.pkl', X_train, Y_train, X_test, Y_test)
        X_train, Y_train, X_test, Y_test = load_data('data.pkl')

        return X_train, X_test, Y_train, Y_test

    def build_and_save_model(self, input_shape, num_classes, model_type='dense'):
        self.model = build_model(input_shape, num_classes, model_type)
        self.original_weights = self.model.get_weights()

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
        client_updates = [res[1] for res in results]

        if not client_updates:
            print("No client updates available for anomaly detection.")
            return super().aggregate_fit(rnd, results, failures)

        # Decrypt client updates
        decrypted_updates = [self._decrypt_params(update.parameters) for update in client_updates]

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
    c_config = config["client1"]
    aes_key = RsaCryptoAPI.load_key('aes_key.bin')  # Load the AES key

    custom_fed_avg = CustomFedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,
        zscore_threshold=server_config["zscore_threshold"],
        momentum=server_config["momentum"],
        aes_key=aes_key
    )

    # Load and prepare data
    X_train, X_test, Y_train, Y_test = custom_fed_avg.load_and_prepare_data(
        dataset_name=c_config["dataset_name"],
        dataset_type=c_config["dataset_type"],
        input_column=c_config["input_column"],
        output_column=c_config["output_column"]
    )

    # Build and save model
    input_shape = X_train.shape[1]
    num_classes = len(np.unique(Y_train))
    custom_fed_avg.build_and_save_model(input_shape, num_classes, model_type=c_config["model_type"])

    # Start the server with the custom strategy
    start_server(
        server_address=server_config["server_address"],
        config=ServerConfig(num_rounds=server_config["num_rounds"]),
        strategy=custom_fed_avg
    )

    # Plot the metrics after training rounds are completed
    plot_training_metrics(rounds, loss, accuracy)

if __name__ == "__main__":
    main()