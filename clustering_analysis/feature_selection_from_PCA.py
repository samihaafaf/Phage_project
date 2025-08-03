
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import argparse
from sklearn import preprocessing

meta_data = pd.read_csv('../data/filtered_meta_data.csv')
k_mer_matrix = pd.read_csv('../data/k_mer_matrix.csv')

feature_data = k_mer_matrix.drop(columns=['id'])




if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Select features that account for 95% of data variation')
    parser.add_argument('num', type=int, help='Number of features to keep from the kmer matrix')
    inputs = parser.parse_args()
    features = inputs.num
    if features > len(feature_data.columns):
        print(f'Invalid input: You requested {features} features, but the k-mer matrix has only {len(feature_data.columns)} features. Please select a number less than or equal to {len(feature_data.columns)}.')
    else:
        print(f'Selecting {features} features from {len(feature_data.columns)}')

        # scales data and applies PCA to reduce dimensionality and find patterns.

        scaled_data = preprocessing.scale(feature_data)
        pca = PCA()
        pca.fit(scaled_data)

        # Cumulative variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumulative_variance >= 0.95) + 1


        top_pcs = pca.components_[:n_components]  

        # (sum of absolute values across top PCs)
        feature_scores = np.sum(np.abs(top_pcs), axis=0)

        top_feature_indices = feature_scores.argsort()[::-1][:features] 

        top_feature_names = feature_data.columns[top_feature_indices]

        selected_cols = ['id'] + list(top_feature_names)
        subset_df = k_mer_matrix[selected_cols]


        subset_df.to_csv("../data/PCA_selected_kmer_matrix.csv", index=False)
