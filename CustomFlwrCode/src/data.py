from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D


def get_model(vocab_size, embedding_dim, input_length, num_classes):
    """Constructs a text classification model."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
        LSTM(128, return_sequences=True),
        GlobalMaxPooling1D(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model_with_progress(model, X_train, Y_train, epochs, batch_size, validation_split=0.2):
    for epoch in tqdm(range(epochs), desc="Training epochs"):
        model.fit(X_train, Y_train, epochs=1, batch_size=batch_size, validation_split=validation_split, verbose=0)