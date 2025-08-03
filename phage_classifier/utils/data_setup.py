import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import numpy as np
#import and merge label data with corresponding host from meta_data csv




def load_and_split_data(model, meta_data, kmer_data, test_size=0.2, random_state=101):
    
    # kmer_data = pd.read_csv(kmer_data)

    # meta_data = pd.read_csv(meta_data)
    # Create a mapping from Accession to Host
    accession_to_host = meta_data.set_index('Accession')['Host']
    kmer_data['Host'] = kmer_data['id'].map(accession_to_host)
   
    X = kmer_data.drop(columns=['id', 'Host']).copy()
    y = kmer_data['Host']
    # Encode host labels for neural-net
    le = LabelEncoder()
    y_nn = le.fit_transform(kmer_data['Host'])
    class_names = np.unique(y)

    if model=='neu-net':
        input_shape = len(X.columns)
        X = torch.from_numpy(X.values).type(torch.float)
        y_nn = torch.from_numpy(y_nn).type(torch.LongTensor)
        output_shape = len(torch.unique(y_nn))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_nn, test_size=test_size, random_state=random_state)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    if model=='neu-net':
        return X_train, X_test, y_train, y_test, output_shape, input_shape, class_names
    else:
        return X_train, X_test, y_train, y_test, class_names

