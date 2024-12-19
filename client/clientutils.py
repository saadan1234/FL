import numpy as np
from flwr.client import Client
from flwr.common import (
    Parameters,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    Code
)
from crypto.rsa_crypto import RsaCryptoAPI
import tensorflow as tf
from model.Modelutils import build_model  # Import the build_model function

def create_flower_client(input_shape, num_classes, model_type, X_train, Y_train, X_test, Y_test):
    """
    Create a Flower client for federated learning.

    Args:
        input_shape: Shape of input data.
        num_classes: Number of output classes.
        model_type: Type of model to build.
        X_train: Training data.
        Y_train: Training labels.
        X_test: Testing data.
        Y_test: Testing labels.

    Returns:
        A Flower client instance.
    """
    class FlowerClient(Client):
        def __init__(self):
            """
            Initialize the Flower client:
            - Load AES key for encryption/decryption.
            - Build and compile the model.
            """
            super().__init__()
            self.aes_key = self.load_key('crypto/aes_key.bin')
            self.decrypted_weights = None
            self.model = build_model(input_shape, num_classes, model_type)

        @staticmethod
        def load_key(filename):
            """
            Load an AES key from a file.

            Args:
                filename: Path to the key file.

            Returns:
                AES key in binary format.
            """
            with open(filename, 'rb') as f:
                return f.read()

        def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
            """
            Encrypt and return the model's parameters.

            Args:
                ins: Instruction to get model parameters.

            Returns:
                Encrypted model parameters.
            """
            print("Getting model parameters for encryption.")
            enc_params = [RsaCryptoAPI.encrypt_numpy_array(self.aes_key, w) for w in self.model.get_weights()]

            return GetParametersRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=Parameters(tensors=enc_params, tensor_type="")
            )

        def set_parameters(self, parameters: Parameters, aes_key: bytes):
            """
            Decrypt and set model parameters.

            Args:
                parameters: Encrypted model parameters.
                aes_key: AES key for decryption.

            Returns:
                Decrypted parameters.
            """
            params = parameters.tensors
            for i, param in enumerate(params):
                decrypted_array = RsaCryptoAPI.decrypt_numpy_array(
                    self.aes_key, param, dtype=self.model.get_weights()[i].dtype
                )
            dec_params = [
                RsaCryptoAPI.decrypt_numpy_array(
                    self.aes_key, param, dtype=self.model.get_weights()[i].dtype
                ).reshape(self.model.get_weights()[i].shape)
                for i, param in enumerate(params)
            ]
            
            self.model.set_weights(dec_params)
            return dec_params

        def fit(self, ins: FitIns) -> FitRes:
            """
            Train the model using provided data and return updated parameters.

            Args:
                ins: Instructions containing encrypted model parameters.

            Returns:
                Fit results including updated parameters.
            """
            self.set_parameters(ins.parameters, self.aes_key)
            self.model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=1, batch_size=32, verbose=1)
            get_param_ins = GetParametersIns(config={'aes_key': self.aes_key})
            return FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=self.get_parameters(get_param_ins).parameters,
                num_examples=len(X_train),
                metrics={}
            )

        def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
            """
            Evaluate the model on the test dataset.

            Args:
                ins: Instructions containing encrypted model parameters.

            Returns:
                Evaluation results including loss and accuracy.
            """
            print("Decrypting model parameters for evaluation.")
            self.set_parameters(ins.parameters, self.aes_key)
            loss, accuracy = self.model.evaluate(X_test, Y_test)
            print(f"Evaluation results - Loss: {loss}, Accuracy: {accuracy}")
            return EvaluateRes(
                status=Status(code=Code.OK, message="Success"),
                loss=loss,
                num_examples=len(X_test),
                metrics={'accuracy': accuracy}
            )

    return FlowerClient()