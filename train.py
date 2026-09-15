"""
Training script for MNIST digit recognition model
"""
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import utils


def build_model():
    """
    Build neural network model for digit classification.
    
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(epochs=15, batch_size=128, validation_split=0.1):
    """
    Load data, build model, and train it.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
    
    Returns:
        Trained model and training history
    """
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    print("\nPreprocessing data...")
    X_train, y_train = utils.preprocess_data(X_train, y_train)
    X_test, y_test = utils.preprocess_data(X_test, y_test)
    
    print("\nBuilding model...")
    model = build_model()
    model.summary()
    
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1
    )
    
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    return model, history, (X_test, y_test)


def save_model(model, model_path='models/mnist_model.h5'):
    """
    Save trained model.
    
    Args:
        model: Trained Keras model
        model_path: Path to save the model
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    # Train the model
    model, history, (X_test, y_test) = train_model(epochs=15, batch_size=128)
    
    # Save the model
    save_model(model)
    
    # Plot training history
    print("\nPlotting training history...")
    utils.plot_training_history(history)
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    y_pred = model.predict(X_test)
    
    # Plot sample predictions
    print("Plotting sample predictions...")
    utils.plot_predictions(X_test, y_test, y_pred, num_samples=10)
    
    # Print classification report
    utils.print_classification_report(y_test, y_pred)
    
    # Plot confusion matrix
    print("Plotting confusion matrix...")
    utils.plot_confusion_matrix(y_test, y_pred)
