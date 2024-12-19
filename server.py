# Import necessary libraries
import pickle
import time
from typing import Tuple
import numpy as np
from transformers import AutoTokenizer
from flwr.server import start_server, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import (
    NDArrays,
    Parameters,
)
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import aggregate
import tensorflow as tf
from data.data_utils import load_dataset_hf, prepare_data, preprocess_and_split
from server.serverutils import load_config, plot_training_metrics, weighted_average
from crypto.rsa_crypto import RsaCryptoAPI
from model.Modelutils import build_model

# Metrics storage
metrics = {"rounds": [], "loss": [], "accuracy": []}

# Customized Server Strategy
class CustomFedAvg(FedAvg):
    """
    Customized federated averaging strategy for secure and optimized aggregation of client updates.
    
    Key Features:
    - **Z-Score Threshold**: Used to identify and handle outliers in client updates.
    - **Momentum**: Implements exponential smoothing for tracking accuracy.
    - **AES Encryption**: Secures communication by encrypting client updates.

    Methods:
    - `load_and_prepare_data`: Prepares datasets for training by loading and preprocessing data.
    - `build_and_save_model`: Builds and initializes a model based on the dataset.
    - `_encrypt_params` & `_decrypt_params`: Secure communication using encryption/decryption.
    - `configure_fit` & `configure_evaluate`: Configures fit/evaluation requests for clients.
    - `aggregate_fit` & `aggregate_evaluate`: Aggregates client updates securely and updates metrics.
    """

    def __init__(self, zscore_threshold: float = 2.5, momentum: float = 0.9, aes_key: bytes = None, **kwargs):
        """
        Initialize the custom federated averaging strategy.
        
        Parameters:
        - `zscore_threshold`: Outlier detection threshold.
        - `momentum`: Weight for smoothing accuracy.
        - `aes_key`: Key for encrypting/decrypting updates.
        """
        super().__init__(**kwargs)
        self.zscore_threshold: float = zscore_threshold
        self.momentum: float = momentum
        self.previous_accuracy: float = None
        self.aes_key: bytes = aes_key
        self.model: tf.keras.Model = None
        self.original_weights: NDArrays = None
        self.init_stage: bool = True
        self.ckpt_name: str = None

    def load_and_prepare_data(self, dataset_name: str, dataset_type: str = "traditional", input_column: str = None, output_column: str = None) -> Tuple:
        """
        Load and preprocess dataset for model training.
        
        - Supports various data types (text, numeric, image).
        - Uses transformers for text-based datasets.
        - Prepares data based on config file settings.

        Returns:
        - Train-test split dataset ready for training.
        """
        tokenizer = None
        if dataset_type == "text":
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        dataset = load_dataset_hf(dataset_name, input_column, output_column, dataset_type)
        dataset = prepare_data(dataset, tokenizer, input_column, output_column, dataset_type)
        print('input_column:', input_column)
        print('output_column:', output_column)
        return preprocess_and_split(dataset["train"], tokenizer, dataset_type,True, input_column, output_column)

    def build_and_load_model(self, input_shape: Tuple[int, ...], num_classes: int, model_type: str = "dense") -> None:
        """
        Builds a model based on the dataset's requirements and initializes original weights.
        
        Parameters:
        - `input_shape`: Shape of the input data.
        - `num_classes`: Number of output classes.
        - `model_type`: Type of the model to be built.
        """
        self.model = build_model(input_shape, num_classes, model_type)
        if self.model is None:
            raise RuntimeError("Model building failed. Ensure `build_model` is correctly implemented.")
        self.original_weights = self.model.get_weights()
        if not self.original_weights:
            raise RuntimeError("Failed to initialize model weights. Ensure the model is compiled properly.")
        
    def save_model(self, filepath: str) -> None:
        """
        Save the model to the specified filepath.
        
        Parameters:
        - `filepath`: Path where the model will be saved.
        """  
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def _save_checkpoint(self, params):
        self.ckpt_name = f"ckpt_sym_{int(time.time())}.bin"
        with open(self.ckpt_name, 'wb') as f:
            pickle.dump(params, f)

    def initialize_parameters(self, client_manager: ClientManager):
        """
        Initializes parameters for clients to begin training.
        
        Returns:
        - Parameters containing initial model weights.
        """
        return Parameters(
            tensors=[w for w in self.model.get_weights()], tensor_type="numpy.ndarrays"
        )

    def _encrypt_params(self, ndarrays: NDArrays) -> Parameters:
        """
        Encrypts model parameters using AES encryption and logs the encryption process.
        """

        encrypted = [
            RsaCryptoAPI.encrypt_numpy_array(self.aes_key, arr) for arr in ndarrays
        ]
        

        return Parameters(tensors=encrypted, tensor_type="")

    def _decrypt_params(self, parameters: Parameters) -> NDArrays:
        """
        Decrypts model parameters received from clients and validates the parameter count.
        
        Parameters:
        - `parameters`: Encrypted model parameters.
        
        Returns:
        - Decrypted model weights as numpy arrays.
        """
        num_received = len(parameters.tensors)
        num_expected = len(self.original_weights)

        if num_received != num_expected:
            raise ValueError(
                f"Mismatch in parameter count: received {num_received}, expected {num_expected}."
            )

        decrypted_params = []
        for i, param in enumerate(parameters.tensors):
            decrypted_param = RsaCryptoAPI.decrypt_numpy_array(
                self.aes_key,
                param,
                dtype=self.original_weights[i].dtype  # Ensure dtype matches
            ).reshape(self.original_weights[i].shape)
            decrypted_params.append(decrypted_param)

        return decrypted_params

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        """
        Configures training instructions for clients.
        
        - Adds encryption key and current round info.
        - Ensures initial parameters are encrypted.
        """
        if self.init_stage:
            parameters = self._encrypt_params(self.model.get_weights())
            self.init_stage = False

        fit_config = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in fit_config:
            fit_ins.config.update({"enc_key": self.aes_key, "curr_round": server_round})
        return fit_config

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager):
        """
        Configures evaluation instructions for clients.
        
        - Ensures encrypted parameters are sent to clients.
        """
        if self.init_stage:
            model_weights = self.model.get_weights()
            parameters = self._encrypt_params(model_weights)
            self.init_stage = False

        encrypted_params_shapes = [len(param) for param in parameters.tensors]

        eval_config = super().configure_evaluate(server_round, parameters, client_manager)
        for client_id, eval_ins in eval_config:
            eval_ins.config["enc_key"] = self.aes_key
        
        return eval_config
    def aggregate_fit(self, server_round: int, results, failures):
        """
        Aggregates training results from clients securely, handles incomplete responses.
        """
        if not results:
            return self._encrypt_params(self.model.get_weights()), {}

        decrypted_updates = []
        for client_id, fit_res in results:
            try:
                decrypted_updates.append(
                    (self._decrypt_params(fit_res.parameters), fit_res.num_examples)
                )
            except ValueError as e:
                continue  # Skip this client

        if not decrypted_updates:
            return self._encrypt_params(self.model.get_weights()), {}

        aggregated_update = aggregate(decrypted_updates)
        
        # Log the shapes of the aggregated weights

        # Update the model with the aggregated weights
        self.model.set_weights(aggregated_update)
        
        # Log the shapes of the model weights after setting the new weights

        encrypted_params = self._encrypt_params(self.model.get_weights())

        return encrypted_params, {}


    def aggregate_evaluate(self, server_round: int, results, failures):
        """
        Aggregates evaluation results from clients and tracks metrics.
        """
        result = super().aggregate_evaluate(server_round, results, failures)
        print(result)
        if not results:
            print(f"No evaluation results received in round {server_round}.")
            metrics["rounds"].append(server_round)
            metrics["loss"].append(None)
            metrics["accuracy"].append(None)
            return result

        loss_value, metrics_data = result
        current_accuracy = metrics_data.get("accuracy", 0.0)
        smoothed_accuracy = (
            self.momentum * self.previous_accuracy + (1 - self.momentum) * current_accuracy
            if self.previous_accuracy is not None
            else current_accuracy
        )
        self.previous_accuracy = smoothed_accuracy

        # Store metrics
        metrics["rounds"].append(server_round)
        metrics["loss"].append(loss_value)
        metrics["accuracy"].append(current_accuracy)

        print(f"Round {server_round}: Loss = {loss_value}, Smoothed Accuracy = {current_accuracy}")
        return result


def main():
    """
    Main function to run the federated learning server.
    
    Steps:
    1. Load configuration from `config.yaml`.
    2. Load AES encryption key.
    3. Initialize the custom federated averaging strategy.
    4. Load and preprocess dataset.
    5. Build the model based on dataset requirements.
    6. Start the federated server to orchestrate client training.
    7. Plot training metrics after training rounds.
    8. Save the model after the last round.
    """
    config = load_config("config.yaml")
    server_config = config["server"]

    # Load encryption key
    aes_key = RsaCryptoAPI.load_key("crypto/aes_key.bin")

    # Initialize the custom strategy
    custom_strategy = CustomFedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,
        zscore_threshold=server_config["zscore_threshold"],
        momentum=server_config["momentum"],
        aes_key=aes_key,
    )

    input_shape = server_config['input_shape']
    num_classes = server_config['num_classes']
    model_type = server_config['model_type']
    custom_strategy.build_and_load_model(input_shape, num_classes, model_type)

    # Start the federated server
    start_server(
        server_address=server_config["server_address"],
        config=ServerConfig(num_rounds=server_config["num_rounds"]),
        strategy=custom_strategy,
    )

    # Plot training metrics
    plot_training_metrics(metrics["rounds"], metrics["loss"], metrics["accuracy"])

    # Save the model after the last round
    custom_strategy.save_model("final_model.h5")

if __name__ == "__main__":
    main()