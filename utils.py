"""
Utility functions for MNIST digit recognition project
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def preprocess_data(X, y=None):
    """
    Preprocess MNIST data by normalizing pixel values and converting labels to one-hot encoding.
    
    Args:
        X: Input features (pixel values 0-255)
        y: Labels (0-9), optional
    
    Returns:
        Normalized X and one-hot encoded y (if provided)
    """
    # Normalize pixel values to 0-1 range
    X = X.astype('float32') / 255.0
    
    # Flatten if needed (for dense network)
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
    
    if y is not None:
        # One-hot encode labels
        from tensorflow.keras.utils import to_categorical
        y = to_categorical(y, 10)
        return X, y
    
    return X


def display_sample_images(X, y, num_samples=10):
    """
    Display sample images from the dataset.
    
    Args:
        X: Input features
        y: Labels
        num_samples: Number of samples to display
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 4))
    axes = axes.flatten()
    
    for i in range(num_samples):
        img = X[i].reshape(28, 28)
        label = np.argmax(y[i]) if len(y[i].shape) > 0 and y[i].sum() > 0 else y[i]
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """
    Plot model training history.
    
    Args:
        history: Keras model history object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_predictions(X, y_true, y_pred, num_samples=10):
    """
    Plot model predictions vs actual labels.
    
    Args:
        X: Input features
        y_true: True labels
        y_pred: Predicted labels
        num_samples: Number of samples to display
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 4))
    axes = axes.flatten()
    
    for i in range(num_samples):
        img = X[i].reshape(28, 28)
        true_label = np.argmax(y_true[i]) if len(y_true[i].shape) > 0 else y_true[i]
        pred_label = np.argmax(y_pred[i]) if len(y_pred[i].shape) > 1 else y_pred[i]
        
        axes[i].imshow(img, cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix for model predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    # Convert one-hot to class labels if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


def print_classification_report(y_true, y_pred):
    """
    Print classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    # Convert one-hot to class labels if needed
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)]))
