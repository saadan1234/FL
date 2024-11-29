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
from clientutils import build_model  # Function to build the required ML model
from crypto.rsa_crypto import RsaCryptoAPI  # Handles RSA encryption/decryption
import tensorflow as tf

def create_gradient_leakage_client(input_shape, num_classes, model_type, X_train, Y_train, X_test, Y_test):
    class GradientLeakageClient(Client):
        def __init__(self):
            super().__init__()
            self.aes_key = self.load_key("crypto/aes_key.bin")
            self.model = build_model(input_shape, num_classes, model_type)

        @staticmethod
        def load_key(filename):
            with open(filename, "rb") as f:
                return f.read()

        def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
            print("Getting model parameters for encryption.")
            enc_params = [
                RsaCryptoAPI.encrypt_numpy_array(self.aes_key, w)
                for w in self.model.get_weights()
            ]
            print(f"Encrypted parameters: {[len(param) for param in enc_params]}")
            return GetParametersRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=Parameters(tensors=enc_params, tensor_type=""),
            )

        def set_parameters(self, parameters: Parameters):
            dec_params = [
                RsaCryptoAPI.decrypt_numpy_array(
                    self.aes_key, param, dtype=self.model.get_weights()[i].dtype
                ).reshape(self.model.get_weights()[i].shape)
                for i, param in enumerate(parameters.tensors)
            ]
            self.model.set_weights(dec_params)

        def fit(self, ins: FitIns) -> FitRes:
            # Set model parameters from the server
            self.set_parameters(ins.parameters)
            self.model.fit(X_train, Y_train, epochs=1, batch_size=32, verbose=1)
            print("Performing Gradient Leakage Attack...")

            optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

            inputs = tf.convert_to_tensor(X_train[:1], dtype=tf.float32)
            targets = tf.convert_to_tensor(Y_train[:1], dtype=tf.int32)

            # Use persistent GradientTape to compute gradients multiple times
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(inputs)
                outputs = self.model(inputs)
                loss = loss_fn(targets, outputs)

            # Compute gradients
            gradients = tape.gradient(loss, self.model.trainable_weights)

            # Call gradient leakage attack with persistent tape
            reconstructed_data, similarity_score = gradient_leakage_attack_keras(
                self.model, gradients, inputs.shape, inputs.numpy()
            )

            # Explicitly delete the tape to free memory
            del tape

            print(f"Similarity Score (MSE): {similarity_score}")

            # Prepare parameters for the server
            get_param_ins = GetParametersIns(config={"aes_key": self.aes_key})
            return FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=self.get_parameters(get_param_ins).parameters,
                num_examples=len(X_train),
                metrics={"similarity_score": float(similarity_score)},
            )


        def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
            print("Decrypting model parameters for evaluation.")
            self.set_parameters(ins.parameters)
            loss, accuracy = self.model.evaluate(X_test, Y_test)
            print(f"Evaluation - Loss: {loss}, Accuracy: {accuracy}")
            return EvaluateRes(
                status=Status(code=Code.OK, message="Success"),
                loss=loss,
                num_examples=len(X_test),
                metrics={"accuracy": accuracy},
            )

    return GradientLeakageClient()


def gradient_leakage_attack_keras(model, gradients, target_shape, actual_data):
    reconstructed_data = tf.Variable(tf.random.normal(target_shape), trainable=True)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    loss_fn = tf.keras.losses.MeanSquaredError()

    for step in range(10):
        with tf.GradientTape() as tape:
            tape.watch(reconstructed_data)
            outputs = model(reconstructed_data)
            # Match gradients with model trainable weights
            loss = tf.add_n([
                loss_fn(tf.convert_to_tensor(g), tf.convert_to_tensor(r)) for g, r in zip(
                gradients,
                tape.gradient(outputs, model.trainable_weights)
            ) if g is not None and r is not None
            ])

        # Compute gradients for reconstruction
        grads = tape.gradient(loss, [reconstructed_data])
        if grads[0] is None:
            raise ValueError("Gradients for reconstructed_data are None. Check graph connections.")

        # Apply gradient updates
        optimizer.apply_gradients(zip(grads, [reconstructed_data]))

        if step % 2 == 0:
            print(f"Step {step}, Loss: {loss.numpy()}")

    reconstructed_data_np = reconstructed_data.numpy()
    similarity_score = tf.reduce_mean(tf.square(actual_data - reconstructed_data_np)).numpy()

    return reconstructed_data_np, similarity_score
