import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tqdm import tqdm

def get_model(input_shape):
    """Constructs a simple model architecture suitable for MNIST."""
    model = Sequential([
    tf.keras.layers.Flatten(input_shape=input_shape),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Custom training function with progress bar
def train_model_with_progress(model, X_train, Y_train, epochs, batch_size, validation_split=0.2):
    for epoch in tqdm(range(epochs), desc="Training epochs"):
        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0)