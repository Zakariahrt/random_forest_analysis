# Income Classification using Random Forest and PCA

This project demonstrates a machine learning pipeline for predicting income categories (`>50K` vs `<=50K`) using the Adult Census Income dataset. It leverages a **Random Forest classifier** and includes a **2D PCA visualization** of the model's decision boundaries.

---

### Dataset Overview

* **Name:** Adult Income Dataset (Census Income)
* **File Used:** `adult_moins.csv`
* **Target Variable:** `income` (binary: `>50K` or `<=50K`)

---

### Project Goals

1. Clean and preprocess real-world census data.
2. Train a **Random Forest** classifier on the complete dataset.
3. Evaluate model performance using metrics and visualization.
4. Apply **Principal Component Analysis (PCA)** to reduce features to 2D.
5. Visualize the model's decision boundary in 2D space.

---

### Requirements

This project uses the following Python libraries:

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install them using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

### Project Workflow

#### 1. **Importing Libraries**

Essential libraries for data handling, preprocessing, modeling, and visualization are imported.

#### 2. **Loading and Cleaning the Dataset**

* The dataset is loaded with predefined column names.
* Missing values (represented as `" ?"`) are detected and removed.
* The `income` target is binarized: `>50K` → 1, `<=50K` → 0.
* Categorical features are encoded using **one-hot encoding**.

#### 3. **Feature Scaling and Splitting**

* Features are scaled using `StandardScaler`.
* The dataset is split into **80% training** and **20% testing** sets.

#### 4. **Training the Random Forest Classifier**

* A `RandomForestClassifier` is trained using the full feature set.
* Performance is evaluated using:

  * Accuracy score
  * Classification report
  * Confusion matrix heatmap

#### 5. **Dimensionality Reduction with PCA**

* The feature set is reduced to 2 principal components using PCA for visualization purposes.
* The reduced dataset is also split into training and test sets.

#### 6. **Training Random Forest on 2D Data**

* A second Random Forest model is trained on the **PCA-reduced 2D data**.

#### 7. **Visualization of Decision Boundary**

* A 2D scatter plot shows the PCA test data.
* A **decision boundary** is overlaid to visualize how the classifier separates the classes in 2D space.

---

### Results & Outputs

* Classification metrics (precision, recall, f1-score)
* Model accuracy
* Confusion matrix (heatmap)
* 2D scatter plot with Random Forest decision boundary

---

### Remarks

* This project highlights the practical use of Random Forest on high-dimensional categorical data.
* **PCA is used for visualization only**, not to improve model performance.
* One-hot encoding of categorical variables is crucial to model performance.

