import tensorflow as tf
from tensorflow.keras import layers, models
from transformers import AutoTokenizer

def build_model(input_shape, num_classes, model_type='dense'):
    """
    Build and compile a Keras model.

    Args:
        input_shape: Shape of the input data (for images or numeric data).
        num_classes: Number of output classes.
        model_type: Type of model to build ('dense', 'image', or 'text').
        vocab_size: Size of the vocabulary (required for 'text' model).

    Returns:
        A compiled Keras model.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    if model_type == 'image':
        print("input_shape", input_shape)
        input_shape = tuple(input_shape)
        
        # Create the model using the Sequential API
        model = models.Sequential()
        model.add(layers.Input(shape=input_shape))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(num_classes, activation='softmax'))
        
        loss = 'sparse_categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy'
    
    elif model_type == 'dense':
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.Dropout(0.2),  # Helps with regularization
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        loss = 'sparse_categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy'
    
    elif model_type == 'text':
        if vocab_size is None:
            raise ValueError("vocab_size must be specified for 'text' models.")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128, input_shape=(input_shape,)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        loss = 'sparse_categorical_crossentropy' if num_classes > 1 else 'binary_crossentropy'
    
    else:
        raise ValueError("Unsupported model type. Choose 'dense', 'image', or 'text'.")

    # Compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=['accuracy']
    )
    return model