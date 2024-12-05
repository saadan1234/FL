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
from client.clientutils import build_model  # Function to build the required ML model
from crypto.rsa_crypto import RsaCryptoAPI  # Handles RSA encryption/decryption
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

def create_gradient_leakage_client(input_shape, num_classes, model_type, X_train, Y_train, X_test, Y_test, data_poisoning, gradient_attack):
    class GradientLeakageClient(Client):
        def __init__(self):
            super().__init__()
            self.aes_key = self.load_key("crypto/aes_key.bin")
            self.model = build_model(input_shape, num_classes, model_type)
            self.previous_gradients = None  # To store cumulative gradients over rounds
            self.reconstructed_data_state = None  # To retain reconstructed data across rounds

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
            # self.model.fit(X_train, Y_train, epochs=1, batch_size=32, verbose=1)
             
            if data_poisoning:

                print(f"X_train, Y_train Type: {type(X_train)} {type(Y_train)}")
                print(f"X_train, Y_train Shape: {X_train[0].shape} {Y_train[0].shape}")
                print(f"X_train, Y_train Length: {len(X_train)} {len(Y_train)}")

                # Poisoning the data
                poisoned_X_train, poisoned_Y_train = poison_data(X_train, Y_train, num_classes)

                print(f"Poisoned X_train, Y_train Type: {type(poisoned_X_train)} {type(poisoned_Y_train)}")
                print(f"PoisonedX_train, Y_train Shape: {poisoned_X_train[0].shape} {poisoned_Y_train[0].shape}")
                print(f"Poisoned X_train, Y_train Length: {len(poisoned_X_train)} {len(poisoned_Y_train)}")

                

                # Train the model with poisoned data
                self.model.fit(poisoned_X_train, poisoned_Y_train, epochs=1, batch_size=32, verbose=1)
                
                # # Generate gradients and simulate gradient leakage
                # inputs = tf.convert_to_tensor(poisoned_X_train[:1], dtype=tf.float32)
                # targets = tf.convert_to_tensor(poisoned_Y_train[:1], dtype=tf.int32)
                
                # with tf.GradientTape() as tape:
                #     outputs = self.model(inputs)
                #     loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(targets, outputs)
                #     gradients = tape.gradient(loss, self.model.trainable_weights)

                # # Poisoned updates sent to the server
                # enc_params = [
                #     RsaCryptoAPI.encrypt_numpy_array(self.aes_key, g.numpy())
                #     for g in gradients
                # ]

                # Prepare parameters for the server
                get_param_ins = GetParametersIns(config={"aes_key": self.aes_key})
                return FitRes(
                    status=Status(code=Code.OK, message="Success"),
                    parameters=self.get_parameters(get_param_ins).parameters,
                    num_examples=len(poisoned_X_train),
                    metrics={},
                )

                # return FitRes(
                #     status=Status(code=Code.OK, message="Success"),
                #     parameters=Parameters(tensors=enc_params, tensor_type=""),
                #     num_examples=len(poisoned_X_train),
                #     metrics={"loss": loss.numpy()},
                # )

            if gradient_attack:

                print("Performing Gradient Leakage Attack...")

                inputs = tf.convert_to_tensor(X_train[:1], dtype=tf.float32)
                targets = tf.convert_to_tensor(Y_train[:1], dtype=tf.int32)

                # Load the tokenizer (make sure to use the same tokenizer that was used for encoding)
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
                # Compute current gradients
                
                with tf.GradientTape() as tape:
                    outputs = self.model(inputs)
                    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(targets, outputs)
                    gradients = tape.gradient(loss, self.model.trainable_weights)

                # Combine current and previous gradients
                if self.previous_gradients is None:
                    combined_gradients = gradients
                else:
                    combined_gradients = [
                        (0.5 * prev + 0.5 * curr) if prev is not None else curr
                        for prev, curr in zip(self.previous_gradients, gradients)
                    ]
            
                # Update the stored gradients
                self.previous_gradients = combined_gradients

                # Call the gradient leakage attack with combined gradients
                reconstructed_data, similarity_score = gradient_leakage_attack(
                    self.model, combined_gradients, inputs.shape, inputs.numpy(), self.reconstructed_data_state
                )
                self.reconstructed_data_state = reconstructed_data

                
                # print(f"Recovered Data: {reconstructed_data}")
                # print(f"Original Data: {inputs}")

                if model_type == "image":

                    # Normalize recovered data to [0, 1] range to match the original scale
                    reconstructed_data_normalized = (reconstructed_data - np.min(reconstructed_data)) / (np.max(reconstructed_data) - np.min(reconstructed_data))

                    # Compute absolute difference
                    difference = np.abs(reconstructed_data_normalized - inputs)

                    # Plot original, recovered, and difference heatmap
                    plt.figure(figsize=(15, 5))

                    # Original data image
                    plt.subplot(1, 3, 1)
                    plt.imshow(inputs[0])  # Show the first sample
                    plt.title("Original Data")
                    plt.axis("off")

                    # Recovered data image
                    plt.subplot(1, 3, 2)
                    plt.imshow(reconstructed_data_normalized[0])  # Show the first sample
                    plt.title("Recovered Data")
                    plt.axis("off")

                    # Difference heatmap
                    plt.subplot(1, 3, 3)
                    plt.imshow(difference[0], cmap="hot")  # Show the first sample
                    plt.title("Difference Heatmap")
                    plt.axis("off")

                    plt.tight_layout()
                    plt.show()

                
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


def gradient_leakage_attack(model, combined_gradients, target_shape, actual_data, initial_state=None):
    if initial_state is not None:
        reconstructed_data = tf.Variable(initial_state, trainable=True)
    else:
        reconstructed_data = tf.Variable(tf.random.normal(target_shape), trainable=True)

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    loss_fn = tf.keras.losses.MeanSquaredError()

    def compute_gradient_loss(tape):
        outputs = model(reconstructed_data)
   
        loss = tf.add_n([
            loss_fn(tf.convert_to_tensor(g), tf.convert_to_tensor(r)) for g, r in zip(
                combined_gradients,
                tape.gradient(outputs, model.trainable_weights)
            ) if g is not None and r is not None
        ])
    
        return loss

    for step in range(1):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(reconstructed_data)
            loss = compute_gradient_loss(tape)

        grads = tape.gradient(loss, [reconstructed_data])
        
        if grads[0] is None:
            print("Warning: Gradients for reconstructed_data are None.")
            break

        optimizer.apply_gradients(zip(grads, [reconstructed_data]))

        if step % 5 == 0:
            print(f"Step {step}, Loss: {loss.numpy()}")

        del tape  # Free memory

    reconstructed_data_np = reconstructed_data.numpy()
    similarity_score = tf.reduce_mean(tf.square(actual_data - reconstructed_data_np)).numpy()

    return reconstructed_data_np, similarity_score

def poison_data(X, Y, num_classes):
    # Example: Label flipping attack
    print(f"Type fo Y : {type(Y)}")
    for i in range(len(Y)):
        Y[i] = (Y[i] + 1) % num_classes  # Simple label flipping
    return X, Y
