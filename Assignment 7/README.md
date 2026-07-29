# Assignment 7 — Customer Segmentation using K-Means Clustering & PCA

## Objective
A shopping mall wants to divide its customers into different groups based on their annual income and spending behavior, in order to run targeted marketing campaigns. This project builds a **K-Means Clustering** model to segment mall customers and applies **Principal Component Analysis (PCA)** to visualize the resulting clusters in two dimensions.

## Dataset
**Mall Customer Segmentation Dataset** (Kaggle)
🔗 https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

The dataset contains 200 records with the following columns:
- `CustomerID`
- `Gender`
- `Age`
- `Annual Income (k$)`
- `Spending Score (1-100)`

> **Note:** The dataset is not included in this repository per the submission guidelines. Please download it from the Kaggle link above and place `Mall_Customers.csv` in the repository root before running the notebook.

## Libraries Used
- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` / `seaborn` — data visualization
- `scikit-learn` — `StandardScaler`, `LabelEncoder`, `KMeans`, `PCA`

## Methodology
1. **Data Understanding** — Loaded the dataset, inspected the first five records, identified numerical (`Age`, `Annual Income`, `Spending Score`) and categorical (`Gender`) features, and reviewed dataset info and summary statistics.
2. **Data Preprocessing** — Checked for missing values (none found), dropped the `CustomerID` column, label-encoded `Gender`, and standardized all numerical features using `StandardScaler`.
3. **Model Development**
   - Applied the **Elbow Method** (K = 1 to 10) to determine the optimal number of clusters, which was identified as **K = 5**.
   - Trained a `KMeans` model with `K = 5` and assigned a cluster label to every customer.
   - Applied **PCA** to reduce the standardized feature set to 2 principal components for visualization.
4. **Visualization and Evaluation**
   - Plotted the Elbow Curve.
   - Plotted a scatter plot of customer clusters (Annual Income vs. Spending Score).
   - Plotted a PCA-based 2D visualization of the clusters.
5. **Conclusion** — Summarized key findings, business applications, a limitation of K-Means, and an advantage of PCA.

## Results
- Optimal number of clusters (from the Elbow Method): **K = 5**
- The two principal components together explain a large majority of the variance in the standardized data, allowing clear 2D visualization of the customer segments.
- Customer segments broadly correspond to combinations of income and spending levels, e.g. high-income/high-spending, high-income/low-spending, low-income/high-spending, low-income/low-spending, and an average-income/average-spending group.

| File | Description |
|---|---|
| `elbow_curve.png` | WCSS vs. number of clusters (Elbow Method) |
| `cluster_scatter.png` | Customer clusters by Annual Income vs. Spending Score |
| `pca_clusters.png` | Customer clusters visualized via PCA (2 components) |

## Conclusion
This project applied K-Means Clustering to segment mall customers into 5 distinct groups based on age, gender, annual income, and spending score, with the optimal cluster count chosen via the Elbow Method. PCA reduced the standardized features to two components, enabling a clear 2D visualization of the segments that closely matched the income-vs-spending view. These segments can help a business design targeted marketing campaigns — for example, loyalty offers for high-value clusters and budget promotions for price-sensitive ones. A key limitation of K-Means is that it requires the number of clusters to be pre-specified and assumes roughly spherical, similarly-sized clusters, which may not always match real customer distributions. A major advantage of PCA is that it reduces dimensionality while preserving most of the variance, making complex data easier to visualize and interpret.

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook Assignment-7.ipynb
```

## Repository Structure
```
.
├── Assignment-7.ipynb   # Main notebook with all tasks (1–5)
├── README.md            # This file
├── elbow_curve.png
├── cluster_scatter.png
└── pca_clusters.png
```
