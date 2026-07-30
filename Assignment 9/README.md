# Cats vs Dogs Classification using CNN — Assignment 9

## Objective
Build a Convolutional Neural Network (CNN) that automatically classifies pet images into two
categories, **Cat** and **Dog**, to help an animal welfare organization automate image sorting.

## Dataset
- **Name:** Cats and Dogs Classification Dataset
- **Source:** [Kaggle — bhavikjikadara/dog-and-cat-classification-dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset)
- **Structure:** `PetImages/Cat/` and `PetImages/Dog/`, ~12,500 images per class (~25,000 total), images named numerically (`0.jpg`, `1.jpg`, ...).
- The dataset is **not included in this repository** — it must be downloaded separately from Kaggle
  and placed alongside the notebook (see *How to Run* below).

## Libraries Used
- TensorFlow / Keras — model building, training, image data generators
- NumPy — numerical operations
- Pillow (PIL) — image loading, dataset cleaning
- Matplotlib / Seaborn — visualizations (sample images, accuracy/loss curves, confusion matrix)
- scikit-learn — evaluation metrics (precision, recall, F1-score, confusion matrix)

## Methodology
1. **Data Understanding** — Explored the folder structure, displayed sample images with labels,
   and identified the number of classes, image dimensions, and total image count.
2. **Data Preprocessing**
   - Removed corrupted/empty image files (a known issue in this dataset).
   - Resized all images to 128 × 128 pixels.
   - Normalized pixel values to the range [0, 1].
   - Split the data into 80% training / 20% testing using Keras' `ImageDataGenerator` with
     `validation_split=0.2`.
3. **Model Development** — Built and trained a CNN (architecture below) for 10 epochs using the
   Adam optimizer and binary cross-entropy loss. Light data augmentation (rotation, shift, zoom,
   horizontal flip) was applied to the training data only, and `EarlyStopping` /
   `ReduceLROnPlateau` callbacks were used as a safety net against overfitting — neither changes
   the architecture nor increases the epoch count beyond 10, they only stop training early or
   lower the learning rate if validation performance stops improving.
4. **Model Evaluation** — Measured test accuracy, precision, recall, and F1-score; plotted the
   confusion matrix and the accuracy/loss curves over epochs.
5. **Conclusion** — Summarized findings, the role of convolution/pooling layers, and CNN
   advantages/limitations.

## CNN Architecture

| Layer | Details |
|---|---|
| Conv2D | 32 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 128 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Flatten | — |
| Dense | 128 neurons, ReLU |
| Dense (Output) | 1 neuron, Sigmoid |

**Compilation:** Optimizer = Adam, Loss = Binary Crossentropy, Metric = Accuracy
**Training:** 10 epochs, batch size 32, input size 128×128×3

## Results

| Metric | Value |
|---|---|
| Test Accuracy | 85.59% |
| Precision | 0.7978 |
| Recall | 0.9536 |
| F1-Score | 0.8688 |

**Confusion Matrix (test set, 4998 images):**

|  | Predicted Cat | Predicted Dog |
|---|---|---|
| **Actual Cat** | 1899 | 600 |
| **Actual Dog** | 125 | 2374 |

The model generalizes well — test accuracy (85.6%) stayed close to, and even slightly above, training
accuracy (84.3%) throughout training, and both train/test loss dropped steadily with no divergence, so
there is **no overfitting**. The model is noticeably better at recognizing Dog images (95% recall) than
Cat images (76% recall), meaning it leans towards predicting "Dog" more often. See the notebook for the
confusion matrix heatmap and the accuracy/loss vs epoch plots.

## Conclusion
This project implemented a CNN to classify cat and dog images, achieving a test accuracy of about
85.6% after just 10 epochs of training on resized, normalized images, with no signs of overfitting
(test accuracy/loss tracked closely with, and even slightly outperformed, training accuracy/loss
throughout). Convolutional layers automatically learn spatial features (edges, textures, shapes)
without manual feature engineering, while pooling layers reduce dimensionality and add robustness to
small spatial shifts. Compared to a standard ANN, a CNN uses far fewer parameters thanks to weight
sharing and preserves spatial structure, making it much better suited to image data. A limitation
observed in this run is a class imbalance in performance — the model recognized Dog images (95%
recall) noticeably better than Cat images (76% recall) — suggesting further gains could come from
more epochs, more balanced/targeted augmentation, or transfer learning (e.g. VGG16, ResNet).

## How to Run
1. Download the dataset from the [Kaggle link](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset) and unzip it.
2. Place the `PetImages` folder (containing `Cat/` and `Dog/`) in the same directory as
   `Assignment-9.ipynb`.
3. Install dependencies:
   ```bash
   pip install tensorflow numpy pillow matplotlib seaborn scikit-learn
   ```
4. Open and run `Assignment-9.ipynb` top to bottom (e.g. via Jupyter Notebook, JupyterLab, or VS Code).

### GPU Training (optional, with automatic CPU fallback)
The notebook automatically detects a GPU and trains on it if available, falling back to CPU
otherwise — no code changes needed.

**For Intel Arc GPUs** (integrated Intel graphics, as opposed to NVIDIA CUDA GPUs): plain
`pip install tensorflow` will **not** see the GPU, since it only supports CUDA out of the box. To
train on an Intel Arc GPU on Windows, install the DirectML plugin instead:
```bash
pip install tensorflow-cpu==2.10.0 tensorflow-directml-plugin
```
This lets TensorFlow use any DirectX 12 GPU, including Intel Arc. If this isn't installed, the
notebook will simply print "No GPU detected" and train on the CPU — everything still runs, just slower.

## Repository Contents
- `Assignment-9.ipynb` — full notebook with code, visualizations, and write-ups for all tasks.
- `README.md` — this file.

> Note: The dataset itself is **not** included in this repository per the assignment instructions;
> refer to the Kaggle link above.
