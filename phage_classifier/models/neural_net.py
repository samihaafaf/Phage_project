import torch
import sys
from pathlib import Path
import torch.nn as nn

current_dir = Path(__file__).resolve().parent
utils_path = current_dir.parent / 'utils'
sys.path.insert(0, str(utils_path))

from helper_functions import accuracy_nn
# cols = 136
# labels = torch.unique(host_label)

class kmer_classifier(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=126):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features = hidden_units, out_features=output_features)
        )
    def forward(self,x):
        return self.linear_layer(x)


# train test ,loop
def train_test_steps(model, X_train,
                     y_train,
                     X_test,
                     y_test,
                     optimizer, loss_fn, epoch=3000):
    epochs = epoch
    results = {'train_loss':[],
                'train_acc':[],
                'test_loss':[],
                'test_acc':[]}

    for each in range(epochs):
        model.train()

        y_logits = model(X_train)
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

        loss = loss_fn(y_logits, y_train)
        train_acc = accuracy_nn(y_true = y_train,y_pred=y_pred)
        
        results['train_acc'].append(train_acc)
        results['train_loss'].append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #testing

        model.eval()

        with torch.inference_mode():

            test_logits = model(X_test)
            test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
            test_loss = loss_fn(test_logits, y_test)

            test_acc = accuracy_nn(y_test, test_pred)

            results['test_acc'].append(test_acc)
            results['test_loss'].append(test_loss)
        
        if epoch>=1000:
            if each % 100 == 0:
                print(f"Epoch: {each} | Loss: {loss:.5f}, Accuracy: {train_acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
        else:
            if each % 10 == 0:
                print(f"Epoch: {each} | Loss: {loss:.5f}, Accuracy: {train_acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

    # print(f'shape of y_test -> {y_test.shape}, shape of pred -> {test_pred.shape}')
    return y_test, test_pred, results

