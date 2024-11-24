import tensorflow as tf

def build_model(input_shape, num_classes, model_type='dense'):
    if model_type == 'dense':
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    elif model_type == 'lstm':
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, input_shape=(input_shape,1), return_sequences=False),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
    else:
        raise ValueError("Unsupported model type. Choose 'dense' or 'lstm'.")
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
