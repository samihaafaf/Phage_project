
from helpers import find_canonical_kmers, compute_kmer_vector
import pandas as pd
from tqdm import tqdm
import time
import argparse

print(f"Loading sequence csv file")
df = pd.read_csv('data/filtered_fasta_data.csv')
# df = df.head(5)

seqs = df['sequence'].tolist()
# seqs = seqs[:5]
print(f"Loaded {len(seqs)} sequences")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute k-mer vectors from sequences')
    parser.add_argument('-k','--k_mer', type=int, default=4, help='Length of k-mers to compute')
    inputs = parser.parse_args()
    k = inputs.k_mer
    print(f'Computing k-mer vectors with k={k}')
    a = find_canonical_kmers(k)

    k_mer_matrix = pd.DataFrame(columns=a)
    start_time = time.time()
    for each in tqdm(range(len(seqs))):
        seq = seqs[each]

        vector = compute_kmer_vector(seq, k, a)
        if vector is None:
            print(f"Error computing k-mer vector for sequence index {each}. Skipping.")
            break
        k_mer_matrix.loc[each] = vector
    end_time = time.time()
    print(f"Time taken to compute k-mer vectors: {end_time - start_time} seconds")
    k_mer_matrix['id'] = df['id']
    k_mer_matrix = k_mer_matrix[['id'] + a]

    k_mer_matrix.to_csv('data/k_mer_matrix.csv', index=False)

    # meta_data = pd.read_csv('filtered_meta_dataV3.csv')
    # if k_mer_matrix.shape[0] == meta_data.shape[0]:
    #     k_mer_matrix['Host'] = meta_data['Host']
    # else:
    #     print(f" k-mer matrix and metadata have different number of rows: {k_mer_matrix.shape[0]} vs {meta_data.shape[0]}. Host column will not be added.")