### Introduction

Phage and bacterial genomes coevolve, leaving patterns in their sequences. Sequence homology remains a widely used approach for identifying known phage–host pairs [1]. Recently, **k-mer based methods** have become popular, as shared oligonucleotide usage between phages and hosts provides useful predictive signals [2].
Recent studies have leveraged various computational methods to infer phage-host relationships from k-mer profiles. This project follows a similar direction, using a subset of viral genome data from [inphared](https://url.au.m.mimecastprotect.com/s/uOKECxng4BI1KD5w5u8fviyNg8O?domain=github.com) to:

- Generate k-mer frequency matrices  
- Perform clustering to explore feature distributions  
- Classify phage hosts using **SVM** and a **neural network**

### Project Structure

This project is organized into **two main components**:

1. **Clustering Analysis** – explores patterns in viral genomes based on k-mer frequency vectors.
2. **Phage Classification** – builds predictive models to classify phage hosts using machine learning and deep learning techniques.

Below is the folder structure:

```
phage_project/
├── clustering_analysis
│   ├── clustering.ipynb
│   ├── explore_group_features.ipynb
│   ├── feature_selection_from_PCA.py
│   └── PCA_analysis.ipynb
├── data
│   ├── 14Apr2025_data_excluding_refseq.tsv
│   ├── data1.csv ...
│   ├── data2.csv ...
├── data_filter.ipynb
├── helpers.py
├── kmer_builder.py
├── phage_classifier
│   ├── models
│   │   ├── kmer_classifier.pth
│   │   └── neural_net.py
│   ├── results
│   │   ├── neural_net
│   │   │   ├── classification_report.png
│   │   │   ├── confusion_matrix.png
│   │   │   ├── loss_curves.png
│   │   │   └── model_architecture.txt
│   │   └── svm
│   │       ├── classification_report.png
│   │       ├── confusion_matrix.png
│   │       └── roc_curve.png
│   ├── train_and_predict.py
│   └── utils
│       ├── data_setup.py
│       └── helper_functions.py
└── requirements.txt
```

### Setup and Dependency Management

1. **Create and activate a virtual environment:**

   - On macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate

     ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt


### Getting started


1. **Download the data files**  
   Place the following files into the `data/` folder (source: [inphared](https://url.au.m.mimecastprotect.com/s/uOKECxng4BI1KD5w5u8fviyNg8O?domain=github.com)):
   - `14Apr2025_genomes_excluding_refseq.fa`
   - `14Apr2025_data_excluding_refseq.tsv`

2. **Run the preprocessing notebook**  
   Open `data_filter.ipynb` and run the cells sequentially to generate:
   - `filtered_fasta_data.csv` — extracted genome sequences from `14Apr2025_genomes_excluding_refseq.fa`  
   - `filtered_meta_data.csv` — cleaned metadata from `14Apr2025_data_excluding_refseq.tsv`

3. Run `kmer_builder.py` script to generate the k-mer frequency vector, which uses `filtered_fasta_data.csv` to create `k_mer_matrix.csv`, stored in the data folder.  
   These files can then be used for downstream clustering analysis and classification. Please refer below for script details.

### Scripts Overview

#### `data_filter.ipynb`

This notebook handles the preprocessing of viral genome data to produce two filtered outputs:

- `filtered_fasta_data.csv` 
- `filtered_meta_data.csv`

The preprocessing steps include:

1. **Host Filtering**: Retains only hosts with ≥200 associated sequences.
2. **Metadata Cleanup**: Removes metadata columns with low informational value or inconsistency, such as:
   - `Jumbophage`
   - `Genbank Division`
   - `Low Coding Capacity Warning`
   - `Isolation Host` (often inconsistent or noisy)
   - `Modification Date` etc
3. **FASTA Filtering**: Extracts only entries labeled as `"complete sequence"` or `"complete genome"` in the FASTA headers.

---

#### `kmer_builder.py`

This script generates k-mer frequency vectors from the filtered genome sequences.

- **Input**: `filtered_fasta_data.csv`
- **Output**: k-mer matrix (e.g., `k_mer_matrix.csv` for `k=4`)
- **Usage**:  
  ```bash
  (myenv) python3 kmer_builder.py -k 5  # builds 5-mer vectors (default is k=4)
  ```


#### `helpers.py`

This module is used in kmer_builder script and provides essential functions for:

- **Canonical k-mer extraction**, following the review [1].
- **Normalization of k-mer vectors**, using a method adapted from an HIV subtyping study [3].
- **Noise reduction** using a **Bloom filter**, which removes rare k-mers (occurring only once) to improve feature quality.


---

### Additional Documentation

- [Clustering Analysis Documentation](https://github.com/samihaafaf/Phage_project/tree/main/clustering_analysis)  
  Details the unsupervised analysis pipeline using k-mer frequency vectors, including PCA, clustering, and feature exploration.

- [Phage Classifier Documentation](https://github.com/samihaafaf/Phage_project/tree/main/phage_classifier)  
  Covers the supervised classification component, including model architecture, training procedures, and evaluation metrics.
 

### References

[1] Edwards, R. A., McNair, K., Faust, K., Raes, J., & Dutilh, B. E. (2016). *Computational approaches to predict bacteriophage--host relationships*. FEMS Microbiology Reviews, 40(2), 258–272.

[2] Roux, S., Hallam, S. J., Woyke, T., & Sullivan, M. B. (2015). *Viral dark matter and virus--host interactions resolved from publicly available microbial genomes*. eLife, 4, e08490.

[3] Solis-Reyes, S., Avino, M., Poon, A., & Kari, L. (2018). *An open-source k-mer based machine learning tool for fast and accurate subtyping of HIV-1 genomes*. PLOS ONE, 13(11), e0206409.
