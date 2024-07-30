import time
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf

# Load the data
with open('data.pkl', 'rb') as f:
    X_train, Y_train, X_test, Y_test = pickle.load(f)

def get_model():
    """Constructs a simple model architecture suitable for MNIST."""
    model = Sequential([
    tf.keras.layers.Flatten(input_shape=X_train.shape[1:]),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = get_model()

# Run the training process 20 times and track the time taken
start_time = time.time()

for i in range(20):    
    # Train the teacher model
    history = model.fit(X_train, Y_train, epochs=25, batch_size=64, validation_split=0.2)

end_time = time.time()
total_time = end_time - start_time
print(f"Total training time: {total_time} seconds")

# Evaluate the teacher model
val_loss, val_mae = model.evaluate(X_test, Y_test)
print(f"Validation MAE: {val_mae}")


# Plot the training loss vs validation loss
plt.figure(figsize=(16, 4))
# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Loss vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Training MAE vs Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.show()