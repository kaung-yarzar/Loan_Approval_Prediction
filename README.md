# Loan_Approval_Prediction

https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset


This project aims to predict loan approval outcomes using machine learning. The following sections detail the workflow.

## 1. Library Imports and Data Loading

- **Importing Core Libraries:**
  - **Pandas, NumPy:** For data manipulation and numerical computations.
  - **Matplotlib & Seaborn:** For data visualization.
  - **SciPy (stats):** For statistical analysis.
  - **missingno:** For visualizing missing data.
  
- **Dataset Loading:**
  - The dataset `data/loan_approval_dataset.csv` is loaded using Pandas.
  - The first three rows are displayed to provide an initial view of the data.
  - The dimensions (number of rows and columns) are printed.
  - A summary of the dataset, including data types and non-null counts, is generated using `df.info`.

## 2. Data Exploration and Visualization

- **Exploratory Data Analysis (EDA):**
  - Visualizations such as histograms and box plots are created to analyze feature distributions.
  - Seaborn is used to generate statistical plots to reveal underlying patterns and relationships.
  - The `missingno` library is utilized to visualize missing data patterns, highlighting data quality issues.

## 3. Data Cleaning and Feature Engineering

- **Handling Missing Values:**
  - Missing data is handled through imputation (using strategies like mean, median, or mode) or by dropping non-critical rows/columns.

- **Encoding Categorical Variables:**
  - Categorical features are transformed into a numerical format using One-Hot Encoding.
  - A `ColumnTransformer` is used to apply appropriate transformations to both categorical and numerical columns.

- **Feature Scaling:**
  - Numerical features are standardized using `StandardScaler` to improve the performance of machine learning algorithms.

## 4. Model Building

- **Model and Pipeline Setup:**
  - Multiple classifiers are imported, including:
    - **K-Nearest Neighbors (KNN)**
    - **Logistic Regression**
    - **Support Vector Machine (SVM)**
    - **Naïve Bayes Variants** (GaussianNB, MultinomialNB, BernoulliNB)
  - A pipeline is created that integrates preprocessing steps (encoding, scaling) with model training.
  
- **Hyperparameter Tuning:**
  - `GridSearchCV` is used to search for the best model parameters via cross-validation.

## 5. Model Evaluation

- **Evaluation Metrics:**
  - **Confusion Matrix:** Displays the counts of true vs. predicted classifications.
  - **Classification Report:** Provides detailed metrics such as precision, recall, F1-score, and support for each class.
  - **Accuracy Score:** Measures the overall proportion of correct predictions.
  - **ROC AUC Score:** Evaluates the model’s ability to distinguish between classes.

