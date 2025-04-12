# utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def plot_curves(train_losses, val_losses, train_accs, val_accs, model_name):
    """
    Plots training and validation loss/accuracy curves.
    
    Args:
        train_losses (list): Training loss for each epoch.
        val_losses (list): Validation loss for each epoch.
        train_accs (list): Training accuracy for each epoch.
        val_accs (list): Validation accuracy for each epoch.
        model_name (str): Name of the model (used in plot titles).
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12,5))
    
    # Plot loss curves
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy curves
    plt.subplot(1,2,2)
    plt.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_normalized_confusion_matrix(cm, classes, title='Normalized Confusion Matrix'):
    """
    Plots a normalized confusion matrix.
    
    Args:
        cm (np.array): Confusion matrix (raw count).
        classes (list): List of class names.
        title (str): Title of the plot.
    """
    # Normalize confusion matrix per true class (row)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10,8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_classification_metrics(all_labels, all_preds, classes):
    """
    Computes per-class precision, recall, and F1-score and plots them as bar charts.
    
    Args:
        all_labels (list or np.array): Ground truth labels.
        all_preds (list or np.array): Predicted labels.
        classes (list): List of class names.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)

    x = np.arange(len(classes))
    width = 0.2  # width of the bars

    plt.figure(figsize=(14,6))
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-score')

    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Per-Class Precision, Recall, and F1-Score')
    plt.xticks(x, classes, rotation=45)
    plt.ylim([0, 1])
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_misclassifications(cm, classes, top_n=5):
    """
    Identify the top-n misclassified classes by comparing off-diagonal sums.
    
    Args:
        cm (np.array): Confusion matrix.
        classes (list): List of class names.
        top_n (int): Number of classes to display.
    """
    # Misclassifications per class (sum of off-diagonals in each row)
    misclassifications = np.sum(cm, axis=1) - np.diag(cm)
    
    # Get indices of top misclassified classes
    top_indices = np.argsort(misclassifications)[-top_n:][::-1]
    
    plt.figure(figsize=(8,5))
    plt.bar([classes[i] for i in top_indices], misclassifications[top_indices])
    plt.xlabel('Classes')
    plt.ylabel('Number of Misclassifications')
    plt.title(f'Top {top_n} Misclassified Classes')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Sample data for testing the plots
    epochs = list(range(1, 11))
    train_losses = np.linspace(1.0, 0.2, num=10)
    val_losses = np.linspace(1.2, 0.3, num=10)
    train_accs = np.linspace(0.5, 0.95, num=10)
    val_accs = np.linspace(0.45, 0.9, num=10)
    model_name = 'TestModel'

    # Test: Plot training curves
    plot_curves(train_losses, val_losses, train_accs, val_accs, model_name)

    # Create a random confusion matrix for testing (for 5 classes)
    classes = ['class1', 'class2', 'class3', 'class4', 'class5']
    cm = np.array([
        [40, 2, 1, 0, 2],
        [3, 35, 3, 2, 1],
        [2, 4, 30, 3, 1],
        [0, 1, 4, 38, 2],
        [1, 0, 2, 3, 44]
    ])

    # Test: Plot normalized confusion matrix
    plot_normalized_confusion_matrix(cm, classes, title='Test Normalized CM')

    # Simulate some predictions for classification metrics
    all_labels = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    all_preds = [0, 1, 2, 2, 4, 0, 2, 2, 3, 3]
    plot_classification_metrics(all_labels, all_preds, classes)

    # Test: Plot misclassified classes
    plot_misclassifications(cm, classes, top_n=3)
