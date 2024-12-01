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
from clientutils import build_model, load_data  # Function to build the required ML model
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

            inputs = tf.convert_to_tensor(X_train[:1], dtype=tf.float32)
            targets = tf.convert_to_tensor(Y_train[:1], dtype=tf.int32)

            # Compute initial loss and gradients
            with tf.GradientTape() as tape:
                outputs = self.model(inputs)
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(targets, outputs)
                gradients = tape.gradient(loss, self.model.trainable_weights)

            # Call gradient leakage attack
            reconstructed_data, similarity_score = gradient_leakage_attack_keras(
                self.model, gradients, inputs.shape, inputs.numpy()
            )

            # Load original data for comparison
            original_data = inputs.numpy()  # Assuming the original data is in the first element
            # Convert recovered data back to original format if necessary
            # Here, we can assume that the original data is in the same shape as reconstructed_data
            recovered_data_converted = reconstructed_data.flatten()  # Adjust this line if needed

            # Create a mapping from numerical values to text
            token_to_text = {i: text for i, text in enumerate(original_data)}

            # Convert recovered data to text format
            recovered_text = [token_to_text.get(int(token), "[UNK]") for token in recovered_data_converted]

            # Print the recovered or generated data in text format
            print("Recovered Data (first 5 elements):", recovered_text[:5])
            print("Original Data (first 5 elements):", original_data[:5])
            print("Similarity Score (MSE):", similarity_score)

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

    def compute_gradient_loss(tape):
        outputs = model(reconstructed_data)
        # Match gradients with model trainable weights
        loss = tf.add_n([
            loss_fn(tf.convert_to_tensor(g), tf.convert_to_tensor(r)) for g, r in zip(
                gradients,
                tape.gradient(outputs, model.trainable_weights)
            ) if g is not None and r is not None
        ])
        return loss

    for step in range(10):
        # Use persistent tape to allow multiple gradient computations
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(reconstructed_data)
            loss = compute_gradient_loss(tape)

        try:
            # Compute gradients for reconstruction
            grads = tape.gradient(loss, [reconstructed_data])
            
            if grads[0] is None:
                print("Warning: Gradients for reconstructed_data are None.")
                break

            # Apply gradient updates
            optimizer.apply_gradients(zip(grads, [reconstructed_data]))

            if step % 2 == 0:
                print(f"Step {step}, Loss: {loss.numpy()}")

        except Exception as e:
            print(f"Error in gradient computation: {e}")
            break
        finally:
            # Explicitly delete the tape to free memory
            del tape

    reconstructed_data_np = reconstructed_data.numpy()
    similarity_score = tf.reduce_mean(tf.square(actual_data - reconstructed_data_np)).numpy()

    return reconstructed_data_np, similarity_score