
### 🧠 Viral Host Classification Using SVM or Neural Network
**Script**: train_and_predict.py

This script trains and tests a **classification model** to predict one of 22 viral host classes using features derived from the k-mer matrix.

It takes the following input files from the `data/` folder:

- `k_mer_matrix.csv` or `PCA_selected_kmer_matrix.csv`
- `filtered_meta_data.csv`

You can choose between two models:

- **SVM** (`svm`)
- **Neural Network** (`neu-net`)

#### Example Usage

Run with **SVM**:

```bash
python3 train_and_predict.py svm
```
Run with **Neural netowork**:

```bash
python3 train_and_predict.py neu-net
```


### ⚙️ Customizable Arguments

You can customize the training process using the following optional arguments:

| Argument           | Type   | Default | Description                                                                 |
|--------------------|--------|---------|-----------------------------------------------------------------------------|
| `model`            | str    | —       | Required. Choose the model: `'svm'` or `'neu-net'`.                         |
| `--test_size`      | float  | 0.2     | Proportion of the dataset to use for testing.                              |
| `--random_state`   | int    | 101     | Random seed for reproducibility.                                           |
| `--hidden_units`   | int    | 128     | Number of hidden units in the neural network. *(Neural net only)*          |
| `--lr`             | float  | 0.001   | Learning rate for the optimizer. *(Neural net only)*       |
| `--epoch`          | int    | 1500    | Number of training epochs. *(Neural net only)*                              |
| `--save_model`     | flag   | False   | If included, saves the trained model to disk, in model folder. *(Neural net only)*          |

> **Note:**
> - Only `--test_size` and `--randoms_state` is used when running the **SVM** model.
> - All other arguments are applicable **only to the neural network**.

### Neural net Architecture

The **neural network architecture** used in this script replicates the model architecture from the study [1],  
with one modification: the number of hidden units is set to **128** to improve classification performance.

---

Functions inside the `utils/` folder handle:

- Preprocessing and setting up data for training.
- Calculating evaluation metrics.
- Saving results to the `results/` folder.

---

### Prediction results

Depending on the selected model, different evaluation outputs are generated and saved to the `results/` directory.

#### ✅ Running with SVM generates:

1. **Confusion Matrix**  
2. **Classification Report**  
3. **ROC Curve** for predictions

#### ✅ Running with Neural Network generates:

1. **Confusion Matrix**  
2. **Classification Report**  
3. **Train-Test Loss Curve**  
4. **Model Architecture Visualization**


## 📈 Prediction Summary

With a test split of **0.2**, the **SVM model** achieves a prediction accuracy of approximately **91%**.  
While SVM is **faster** in terms of prediction time, its performance can be **surpassed** by the neural network when trained for more than **7000 epochs** with **128 hidden units** using the default configuration.

The **neural network** shows a **stable learning behavior**, as observed in the smooth train-test loss curves. Its performance and efficiency can be further improved by:

- Modifying the **network architecture**
- **Tuning hyperparameters** such as learning rate, hidden units, and epochs

Overall, the neural network holds greater potential for improved classification accuracy with proper optimization.


### References
[1] Çi̇ftçi̇, B., & Teki̇n, R. (2024). Prediction of viral families and hosts of single-stranded RNA viruses based on K-Mer coding from phylogenetic gene sequences. Computational Biology and Chemistry, 112, 108114.


