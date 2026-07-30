# Assignment 8 — Handwritten Digit Classification using Artificial Neural Network (ANN)

## Objective
A postal service organization wants to automate the recognition of handwritten digits on postal codes. This project builds an **Artificial Neural Network (ANN)** using TensorFlow/Keras to classify handwritten digits (0-9) from the MNIST dataset.

## Dataset
**MNIST Handwritten Digits Dataset** (Kaggle, CSV format)
🔗 https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

The dataset provides `mnist_train.csv` (60,000 images) and `mnist_test.csv` (10,000 images). Each row represents one 28×28 grayscale image, flattened into 784 pixel columns (`1x1` ... `28x28`), with a `label` column (0-9) indicating the digit.

> **Note:** The dataset is not included in this repository per the submission guidelines. Please download `mnist_train.csv` and `mnist_test.csv` from the Kaggle link above and place them in the repository root before running the notebook.

## Libraries Used
- `pandas`, `numpy` — data loading and manipulation
- `matplotlib`, `seaborn` — data visualization
- `scikit-learn` — `train_test_split`, `confusion_matrix`, `classification_report`
- `tensorflow` / `keras` — building, training, and evaluating the ANN

## Methodology
1. **Data Understanding** — Loaded `mnist_train.csv` and `mnist_test.csv`, inspected the first five records, identified the 784 pixel columns as input features and `label` as the target variable, reviewed dataset dimensions, and visualized sample handwritten digits.
2. **Data Preprocessing**
   - Checked for missing values (none found).
   - Combined the Kaggle train/test files and re-split them into an **80% train / 20% test** split (as required by the assignment), using stratified sampling to preserve class balance.
   - Separated pixel features (`X`) from the digit label (`y`).
   - Normalized all pixel values from the 0-255 range to **0-1**.
   - One-hot encoded the digit labels into a 10-dimensional categorical format.
3. **Model Development** — Built and trained a fully-connected ANN (architecture below) using the Adam optimizer, categorical crossentropy loss, and accuracy as the evaluation metric, for **10 epochs**.
4. **Model Evaluation** — Evaluated the trained model on the held-out test set using test accuracy, a confusion matrix, and a full classification report, and plotted accuracy/loss curves across epochs.
5. **Conclusion** — Summarized key findings, the role of hidden layers, an advantage of deep learning over traditional ML, and a limitation of ANNs.

## Model Architecture
| Layer | Type | Units | Activation |
|---|---|---|---|
| Input | — | 784 (28×28 flattened pixels) | — |
| Hidden Layer 1 | Dense | 128 | ReLU |
| Hidden Layer 2 | Dense | 64 | ReLU |
| Output Layer | Dense | 10 | Softmax |

**Compilation settings:**
- Optimizer: `Adam`
- Loss: `categorical_crossentropy`
- Metric: `accuracy`
- Epochs: `10` | Batch size: `128`

## Results
- **Test Accuracy:** 97.33%
- **Test Loss:** 0.1012
- Training and validation accuracy/loss curves show steady improvement across epochs with no significant overfitting within 10 epochs.
- The confusion matrix shows the model performs consistently well across all 10 digit classes, with most errors occurring between visually similar digits (e.g., 4/9, 3/5, 7/1).

| File | Description |
|---|---|
| `sample_digit.png` | Example handwritten digit from the dataset |
| `sample_digits_grid.png` | One example image per digit class (0-9) |
| `label_distribution.png` | Class balance across the training set |
| `accuracy_vs_epoch.png` | Training vs. validation accuracy over 10 epochs |
| `loss_vs_epoch.png` | Training vs. validation loss over 10 epochs |
| `confusion_matrix.png` | Confusion matrix of predictions on the test set |
| `sample_predictions.png` | Sample test images with predicted vs. actual labels |

## Conclusion
This project developed an Artificial Neural Network using TensorFlow/Keras to classify handwritten digits from the MNIST dataset, achieving a test accuracy of 97.33% after 10 training epochs with a simple two-hidden-layer architecture. Hidden layers are essential because they let the network learn increasingly abstract, non-linear feature combinations from raw pixel data — without them, the model could only represent simple linear relationships and would fail to capture the complex stroke patterns that distinguish digits. A key advantage of deep learning over traditional machine learning is that it can automatically learn relevant features directly from raw data, removing the need for manual feature engineering. However, a notable limitation of ANNs is that they typically require large labeled datasets and careful tuning to avoid overfitting or slow convergence. This approach directly supports real-world postal automation by enabling fast, accurate reading of handwritten postal codes.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow jupyter
jupyter notebook Assignment-8.ipynb
```

## Repository Structure
```
.
├── Assignment-8.ipynb        # Main notebook with all tasks (1-5)
├── README.md                 # This file
├── sample_digit.png
├── sample_digits_grid.png
├── label_distribution.png
├── accuracy_vs_epoch.png
├── loss_vs_epoch.png
├── confusion_matrix.png
└── sample_predictions.png
```
