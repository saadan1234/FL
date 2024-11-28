import time
import numpy as np
from crypto.rsa_crypto import RsaCryptoAPI

class Parameters:
    def __init__(self, tensors, tensor_type):
        self.tensors = tensors
        self.tensor_type = tensor_type

class TestEncryption:
    def __init__(self, aes_key=None):
        if aes_key is None:
            self.__aes_key = RsaCryptoAPI.gen_aes_key()
        else:
            self.__aes_key = aes_key

    def _encrypt_params(self, ndarrays):
        enc_tensors = [RsaCryptoAPI.encrypt_numpy_array(self.__aes_key, arr) for arr in ndarrays]
        return Parameters(tensors=enc_tensors, tensor_type="")

    def _decrypt_params(self, parameters):
        return [RsaCryptoAPI.decrypt_numpy_array(self.__aes_key, param, dtype=np.float64).reshape((10, 10)) for param in parameters.tensors]

    def save_key(self, filename):
        with open(filename, 'wb') as f:
            f.write(self.__aes_key)

    @staticmethod
    def load_key(filename):
        with open(filename, 'rb') as f:
            return f.read()

# Generate a small dummy numpy array
dummy_array = np.random.rand(10, 10)  # Example: 10x10 array
print("Original array:")
print(dummy_array)

# Initialize the TestEncryption class and save the AES key
test_encryption = TestEncryption()
test_encryption.save_key('aes_key.bin')

# Load the AES key from the file
loaded_aes_key = TestEncryption.load_key('aes_key.bin')

# Initialize the TestEncryption class with the loaded AES key
test_encryption_with_loaded_key = TestEncryption(aes_key=loaded_aes_key)

# Measure encryption time
start_time = time.time()
encrypted_params = test_encryption_with_loaded_key._encrypt_params([dummy_array])
encryption_time = time.time() - start_time
print(f"Encryption time: {encryption_time} seconds")

# Measure decryption time
start_time = time.time()
decrypted_array = test_encryption_with_loaded_key._decrypt_params(encrypted_params)
decryption_time = time.time() - start_time
print(f"Decryption time: {decryption_time} seconds")

# Verify that the decrypted array matches the original array
if np.array_equal(dummy_array, decrypted_array[0]):
    print("Decryption successful, arrays match.")
else:
    print("Decryption failed, arrays do not match.")

print("Decrypted array:")
print(decrypted_array[0])