## PCA and Clustering of k-mer Features

This section performs PCA and clustering on k-mer frequency vectors to identify major axes of variation and explore meaningful patterns in the dataset.

---

### 1. Principal Component Analysis (PCA)
**Script:** `PCA_analysis.ipynb`

- Performs PCA on the k-mer matrix to identify the major axes of variation.
- Highlights top k-mers contributing most to the first principal component (PC1).
- Selects important features based on cumulative contributions across all principal components that explain 95% of total variance [1].
- Visualization includes scree plot and feature loadings.

### 2. Optional: Select Top *n* Features from k-mer Matrix

You may optionally choose the top *n* features from the original k-mer feature matrix to reduce dimensionality. This can potentially **improve prediction performance**, especially when dealing with a large number of features [1].

**Script:** `feature_selection_from_PCA.py`

This script performs feature selection on the original `k_mer_matrix.csv` using **Principal Component Analysis (PCA)**. The process includes:

- Normalizing the data.
- Applying PCA to determine the minimum number of components that explain at least **95%** of the total variance.
- Ranking features based on their **aggregate contribution** (absolute loadings) across the selected components.
- Selecting the top *n* features as specified by the user.
- Saving the reduced matrix as `PCA_selected_kmer_matrix.csv` in the `data/` folder.

This method follows the approach by **Song et al. (2010)**, using PCA not just for dimensionality reduction but also as an embedded feature selection technique.

Run the script using the following command:

```bash
python3 feature_selection_from_PCA.py <num>  #num = number of features to select
```

### 3. Clustering
**Script:** `clustering.ipynb`

This notebook performs **clustering** on a k-mer feature matrix, which can be either:

- `k_mer_matrix.csv` (full feature set), or  
- `PCA_selected_kmer_matrix.csv` (reduced set from PCA-based selection)

#### Steps:
1. Generates an **elbow plot** to determine the optimal number of clusters using **Within-Cluster Sum of Squares (WCSS)**.
2. Applies **K-Means clustering** using the chosen number of clusters.
3. Assigns each sample a **cluster label**.
4. Saves the clustered dataset in the `data/` directory for downstream analysis.

---

### 4. Exploratory feature analysis
**Script:** `explore_group_features.ipynb`

This notebook investigates how **biological features vary across clusters**, using either:

- `all_data_cluster.csv`, or  
- `pca_data_clustered.csv`

#### Statistical Methods:

- **Continuous features**: Tested using **ANOVA** and effect size measured with **Eta Squared (η²)**.
- **Categorical features**: Tested using **Chi-square test** and effect size measured with **Cramér’s V**.

#### 📈 Visualizations:
- Plots showing how features are distributed across clusters.

#### 🧠 Key Findings:

- Strong differences observed in categorical features like `Host`, `Class`, and `Phylum`.
- Significant variation in continuous features such as `molGC (%)` and `Genome Length`.
- **GC content** and **Host** features are the most distinctive, reflecting differences in k-mer profiles across groups.



---
**Reference:**

[1]Song, Fengxi, Zhongwei Guo, and Dayong Mei. "Feature selection using principal component analysis."  
 
