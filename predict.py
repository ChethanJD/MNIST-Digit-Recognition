"""
Prediction script for MNIST digit recognition model
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
import utils


def load_model(model_path='models/mnist_model.h5'):
    """
    Load trained model.
    
    Args:
        model_path: Path to the saved model
    
    Returns:
        Loaded Keras model
    """
    model = keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model


def predict_on_test_set(model, num_samples=20):
    """
    Make predictions on MNIST test set and visualize results.
    
    Args:
        model: Trained Keras model
        num_samples: Number of test samples to predict on
    """
    print("Loading MNIST test set...")
    (_, _), (X_test, y_test) = mnist.load_data()
    
    # Preprocess test data
    X_test = utils.preprocess_data(X_test)
    
    print(f"\nMaking predictions on {num_samples} test samples...")
    X_test_sample = X_test[:num_samples]
    y_test_sample = y_test[:num_samples]
    
    y_pred = model.predict(X_test_sample)
    
    # Display predictions
    print("\nPredictions:")
    for i in range(num_samples):
        pred_label = np.argmax(y_pred[i])
        confidence = np.max(y_pred[i]) * 100
        print(f"Sample {i+1}: Predicted={pred_label}, Actual={y_test_sample[i]}, Confidence={confidence:.2f}%")
    
    # Visualize predictions
    print("\nVisualizing predictions...")
    utils.plot_predictions(X_test_sample, y_test_sample, y_pred, num_samples=min(10, num_samples))


def predict_custom_digit(model, image_array):
    """
    Make prediction on a custom digit image.
    
    Args:
        model: Trained Keras model
        image_array: 28x28 numpy array representing the digit
    
    Returns:
        Predicted label and confidence
    """
    # Normalize and flatten
    image = image_array.astype('float32') / 255.0
    image = image.reshape(1, -1)
    
    # Make prediction
    y_pred = model.predict(image, verbose=0)
    pred_label = np.argmax(y_pred[0])
    confidence = np.max(y_pred[0]) * 100
    
    print(f"Predicted: {pred_label}, Confidence: {confidence:.2f}%")
    return pred_label, confidence


if __name__ == "__main__":
    # Load the trained model
    model = load_model()
    
    # Make predictions on test set
    predict_on_test_set(model, num_samples=20)
    
    print("\n" + "="*50)
    print("Prediction script completed!")
