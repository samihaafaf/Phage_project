
import pandas as pd
import torch
from torch import nn
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
from utils import data_setup
from utils import helper_functions as hf
import numpy as np
from models import neural_net
from sklearn import svm

# setting train test data
meta_data = pd.read_csv('../data/filtered_meta_data.csv')
kmer_data = pd.read_csv('../data/k_mer_matrix.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and predict viral hosts using k-mer matrix' \
    'data using either a SVM or neural network architecture')
    parser.add_argument('model', type=str, help='Select the model for prediction',
                         choices=['neu-net','svm'])
    parser.add_argument('--test_size', type=float, default=0.2, help='Select test size for training')
    parser.add_argument('--random_state', type=int, default=101, help='Select random state')
    parser.add_argument('--hidden_units', type=int, default=128, help='Select number of hidden' \
    'nodes in neural network')
    parser.add_argument('--lr', type=float, default=0.001, help='Select learning rate for ' \
    'optimizer used in the neural net')
    parser.add_argument('--epoch', type=int, default=1500, help='Select epochs for training') 
    parser.add_argument('--save_model', action='store_true', help='Save the trained model to disk')

    inputs = parser.parse_args()
    model = inputs.model
    test_size = inputs.test_size
    random_state = inputs.random_state
    hidden_units = inputs.hidden_units
    epochs = inputs.epoch
    lr = inputs.lr
    if model=='neu-net':
        device = "cude" if torch.cuda.is_available() else "cpu"
        torch.manual_seed(42)

        X_train, X_test, y_train, y_test, output_shape, input_shape, class_names = data_setup.load_and_split_data(
            model, meta_data, kmer_data, test_size=test_size, random_state=random_state)
        

        # argparse for save model if want

        loss_fn = nn.CrossEntropyLoss()


        classifier = neural_net.kmer_classifier(input_features=input_shape, output_features=output_shape, 
                                        hidden_units=hidden_units).to(device)
        optimizer = torch.optim.Adam(params=classifier.parameters(),
                                    lr=lr)
        y_test, test_pred, results = neural_net.train_test_steps(classifier, X_train.to(device),
                            y_train.to(device),
                            X_test.to(device),
                            y_test.to(device),
                            optimizer=optimizer, loss_fn=loss_fn, epoch=epochs)
        if inputs.save_model:
            torch.save(classifier.state_dict(), './models/kmer_classifier.pth')


        hf.plot_loss_curves(results)
        hf.save_model_architecture(classifier, X_train[0].shape)
        hf.plot_and_save_confusion_matrix(model, y_test.cpu().detach().numpy(), test_pred.cpu().detach().numpy(), class_names)
        hf.save_classification_report(model,y_test.cpu().detach().numpy(), test_pred.cpu().detach().numpy(), np.unique(y_test), class_names)


    else:
        X_train, X_test, y_train, y_test, class_names = data_setup.load_and_split_data(
            model, meta_data, kmer_data, test_size=test_size, random_state=random_state)
        # 
        mod = svm.SVC(probability=True)
        mod.fit(X_train, y_train)

        y_pred = mod.predict(X_test)
        hf.plot_and_save_confusion_matrix( model, y_test, y_pred, class_names)
        hf.save_classification_report( model, y_test, y_pred, class_names)
        hf.plot_multiclass_roc(mod, X_test, y_test, class_names)
        

        




