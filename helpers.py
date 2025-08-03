# helpers.py
from itertools import product
from bitarray import bitarray
import mmh3  

class BloomFilter:
    def __init__(self, size, num_hashes):
        self.size = size
        self.array = bitarray(size)
        self.array.setall(0)
        self.num_hashes = num_hashes
        

    def get_hashes(self, k_mer):
        hashes = []
        for i in range(self.num_hashes):
            hash_value = mmh3.hash(k_mer, i) % self.size
            hashes.append(hash_value)
        return hashes


    def add(self, k_mer):
        for hash_val in self.get_hashes(k_mer):
            self.array[hash_val] = 1

    def exists(self, k_mer):
        return all(self.array[hash_val] for hash_val in self.get_hashes(k_mer))


# bf = BloomFilter(size=17000, num_hashes=4)

def reverse_complement(seq):
    pairs = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
    seq = seq.upper()
    try:
        return ''.join(pairs[base] for base in reversed(seq))
    except KeyError:
        return ''

def find_canonical_kmers(k):
    seen = set()
    kmers = []
    for p in product("ACGT", repeat=k):
        kmer = ''.join(p)
        rc = reverse_complement(kmer)
        canonical = min(kmer, rc)
        if canonical not in seen:
            seen.add(canonical)
            kmers.append(canonical)
    return sorted(kmers)
def compute_kmer_vector(seq, k, canonical_kmers):
    total_kmers = 0
    counts = {kmer: 0 for kmer in canonical_kmers}
    bf = BloomFilter(size=17000, num_hashes=4)
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if any(base not in 'ATGC' for base in kmer):
            continue
        rc = reverse_complement(kmer)
        canonical = min(kmer, rc)
        
        if not bf.exists(canonical):

            bf.add(canonical)
        else:
            counts[canonical] += 1
            total_kmers += 1
        

    # Normalizing frequency vector
    vector = [counts[kmer] / total_kmers if total_kmers > 0 else 0 for kmer in canonical_kmers]
    return vector
    # return counts

