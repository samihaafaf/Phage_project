import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from torchinfo import summary
import os


def save_model_architecture(model, input_size, save_path="./results/neural_net/model_architecture.txt"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write(str(model))  
        f.write("\n\nDetailed Summary:\n")
    # print(str(model))
        try:
            model_cpu = model.cpu()
            summary_str = summary(model_cpu, input_size) 
            # print(summary_str)
            f.write(str(summary_str))
        except Exception as e:
            f.write(f"\nCould not generate detailed summary: {e}")


# calculates accuracy of neural net model
def accuracy_nn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc



def plot_multiclass_roc(model, X_test, y_test, class_names, save_path='./results/svm/roc_curve.png'):
    """
    Plots ROC curves for a multiclass classifier.

    Parameters:
        model: Trained classifier with predict_proba method.
        X_test: Test features
        y_test: True labels
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    y_score = model.predict_proba(X_test)
    class_names = class_names
    y_test_bin = label_binarize(y_test, classes=class_names)
    n_classes = len(class_names)

    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()



    plt.savefig(save_path, dpi=300)

    plt.close()



# adjust for neural network as well
def plot_and_save_confusion_matrix(model,y_true, y_pred, class_names,title='Confusion Matrix'):
    """
    Plots and saves the confusion matrix.
    
    Parameters:
        y_true (array-like): Ground truth labels
        y_pred (array-like): Predicted labels
    """
    # class_names = np.unique(y_true)
    # print(f'class names are {class_names}')
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap="crest", linewidth=.5,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    if model=='svm':
        save_path = './results/svm/confusion_matrix.png'
       
    else:
        save_path = './results/neural_net/confusion_matrix.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)

    plt.close()


# adjust for nrural network as well
def save_classification_report(model, y_true, y_pred, labels, class_names=None):
    """
    Generates classification metrics and saves them as an image table.

    Parameters:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        class_names (list or array): Unique class names
        save_path (str): Path to save the table image
    """
    # print(f'ytrue type -> {type(y_true)}, ypred type -> {type(y_pred)}')
    # print(f'ytrue shape -> {y_true.shape}, ypred shape -> {y_pred.shape}')
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    if model=='svm':
        df = pd.DataFrame({
            'Class': labels,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })
    else:
        df = pd.DataFrame({
            'Class': class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })


    df.loc[len(df)] = [ 'Average',precision.mean(), recall.mean(), f1.mean(), support.sum()]
    
    # adding accuracy
    acc = accuracy_score(y_true, y_pred)
    df.loc[len(df)] = ['Accuracy', acc, np.nan, np.nan, np.nan]

    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.5 * len(df))) 
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center', colColours=["#d0e1f9"]*len(df.columns))
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 2.0) 

    plt.tight_layout()
    if model=='svm':
        save_path = './results/svm/classification_report.png'
        
    else:
        save_path = './results/neural_net/classification_report.png'
        # plt.savefig('./results/neural_net/classification_report.png', dpi=300)
    # plt.show()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_loss_curves(results, save_path="./results/neural_net/loss_curves.png"):
    """Plots training curves and saves them as a single image.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
        save_path (str): path to save the plot image
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Convert tensors to numpt  list
    def to_numpy_list(tensor_list):
        return [x.detach().cpu().numpy() if hasattr(x, 'detach') else x for x in tensor_list]

    loss = to_numpy_list(results["train_loss"])
    test_loss = to_numpy_list(results["test_loss"])
    accuracy = to_numpy_list(results["train_acc"])
    test_accuracy = to_numpy_list(results["test_acc"])

    epochs = range(len(loss))

    plt.figure(figsize=(15, 7))

    # Plot loss and accuracies
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()




