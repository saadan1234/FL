import pickle
import time
from typing import Dict, List, Optional, Tuple, Union
from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAvg
import numpy as np
from serverutils import detect_anomalies_zscore, load_config, plot_training_metrics, weighted_average
from crypto.rsa_crypto import RsaCryptoAPI
from transformers import AutoTokenizer
from model import build_model
from clientutils import load_data, load_dataset_hf, prepare_data, preprocess_and_split, save_data
from flwr.common import (
    EvaluateRes,
    NDArrays,
    Scalar,
    Parameters,
    FitIns,
    FitRes,
    EvaluateIns,
    MetricsAggregationFn)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate

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
        self.init_stage = True
        self.ckpt_name = None
    

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

    def initialize_parameters(self, client_manager: ClientManager):
        #log(INFO, f'Server AES key: {self.__aes_key}')
        # TODO: Save initial checkpoint
        return Parameters(
            tensors=[w for w in self.model.get_weights()],
            tensor_type="numpy.ndarrays")

    def _get_param_info(self):
        return zip([],
                    [v.shape for v in self.model.get_weights()],
                    [v.dtype for v in self.model.get_weights()])
    
    def _decrypt_params(self, parameters: Parameters) -> NDArrays:
        params = parameters.tensors
        dec_params = [RsaCryptoAPI.decrypt_numpy_array(self.aes_key, param, dtype=self.original_weights[i].dtype).reshape(self.original_weights[i].shape) for i, param in enumerate(params)]
        return dec_params

    def _encrypt_params(self, ndarrays: NDArrays) -> Parameters:
        enc_tensors = [RsaCryptoAPI.encrypt_numpy_array(self.aes_key, arr) for arr in ndarrays]
        return Parameters(tensors=enc_tensors, tensor_type="")

    def _save_checkpoint(self, params):
        self.ckpt_name = f"ckpt_sym_{int(time.time())}.bin"
        with open(self.ckpt_name, 'wb') as f:
            pickle.dump(params, f)

    def _load_previous_checkpoint(self):
        with open(self.ckpt_name, 'rb') as f:
            params = pickle.load(f)
        return params

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
            ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        if self.init_stage:
            # encrypt all params
            parameters = self._encrypt_params(parameters.tensors)
            self._save_checkpoint(parameters)
            self.init_stage = False

        if len(parameters.tensors) == 0:
            parameters = self._load_previous_checkpoint()

        fit_config = super().configure_fit(server_round, parameters, client_manager)

        for _, fit_ins in fit_config:
            fit_ins.config['enc_key'] = self.aes_key
            fit_ins.config['curr_round'] = server_round

        return fit_config
    
    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
            ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        if self.init_stage:
            parameters = self._encrypt_params(parameters.tensors)
            self.init_stage = False

        eval_config = super().configure_evaluate(server_round, parameters, client_manager)
        for _, ins in eval_config:
            ins.config['enc_key'] = self.aes_key
        return eval_config
    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None

        # We deserialize using our custom method
        if self.init_stage:
            parameters_ndarrays = parameters.tensors
        else:
            parameters_ndarrays = self._decrypt_params(parameters)

        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics
    

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results and detect anomalies."""
        # Collect client updates for anomaly detection
        # client_updates = [res[1] for res in results]

        # if not client_updates:
        #     print("No client updates available for anomaly detection.")
        #     return super().aggregate_fit(server_round, results, failures)

        # Decrypt client updates
        decrypted_updates = [
            (self._decrypt_params(fit_res.parameters), fit_res.num_examples)
            for i, (_, fit_res) in enumerate(results) 
        ]

        # updates_matrix = np.array([np.concatenate([p.flatten() for p in update]) for update in decrypted_updates])

        # if updates_matrix.size == 0:
        #     print("Empty updates matrix, skipping anomaly detection.")
        #     return super().aggregate_fit(server_round, results, failures)

        # # Detect anomalies using Z-score
        # zscore_anomalies = detect_anomalies_zscore(updates_matrix, self.zscore_threshold)

        # # Exclude detected anomalies from aggregation
        # filtered_results = [res for i, res in enumerate(results) if i not in zscore_anomalies]

        # print(f"Round {server_round}: Detected {len(zscore_anomalies)} anomalous clients, excluded from aggregation.")

        # # Apply momentum and aggregate filtered results
        # filtered_updates = np.array([np.concatenate([p.flatten() for p in res.parameters]) for res in filtered_results])
        # aggregated_update = np.mean(filtered_updates, axis=0)

        # if self.previous_update is not None:
        #     aggregated_update = self.momentum * self.previous_update + (1 - self.momentum) * aggregated_update

        # self.previous_update = aggregated_update

        # # Reconstruct update parameters
        # aggregated_update_params = [
        #     np.reshape(aggregated_update[i:i + len(layer)], layer.shape)
        #     for i, layer in enumerate(self.model.get_weights())
        # ]

        # Encrypt the aggregated update parameters
        enc_aggregated_update_params = self._encrypt_params(aggregate(decrypted_updates))

        return enc_aggregated_update_params, {}
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation results and detect anomalies."""
        aggregated_result = super().aggregate_evaluate(server_round, results, failures)

        # Track round data
        rounds.append(server_round)
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
            print(current_accuracy,'this is current acuracy')

            # Store smoothed metrics
            loss.append(loss_value)
            accuracy.append(smoothed_accuracy)

        print(f"Round {server_round} - Loss: {loss_value}, Smoothed Accuracy: {accuracy[-1]}")

        return aggregated_result



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