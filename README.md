# Loan_Approval_Prediction

https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset

1. Library Imports and Data Loading
Importing Core Libraries:
The notebook starts by importing essential Python libraries for data analysis and visualization:

Pandas: For reading the dataset and handling data frames.
NumPy: For numerical computations.
Matplotlib & Seaborn: For plotting graphs and visualizing data distributions.
Scipy.stats: Likely used for statistical analysis and outlier detection.
missingno: For visualizing missing data patterns.
Loading the Dataset:
The dataset is loaded using pd.read_csv("data/loan_approval_dataset.csv"). After loading:

The first three rows of the dataset are displayed (using df.head(3)) to give a quick glance at the data.
The shape (number of rows and columns) is printed to show the dataset dimensions.
A summary of the DataFrame’s information (data types, non-null counts) is printed using df.info.
2. Data Exploration and Visualization
Exploratory Data Analysis (EDA):
The code performs EDA by:

Generating various plots (e.g., histograms, box plots) to visualize the distributions of numerical features.
Using Seaborn for more sophisticated statistical plots to uncover patterns and correlations between features.
Leveraging missingno to create visual matrices or bar charts that highlight missing values across different columns, which helps in understanding the quality and completeness of the dataset.
Preliminary Statistical Analysis:
The notebook likely includes basic statistical measures (mean, median, mode, standard deviation) to understand the central tendencies and spread of the data.

3. Data Cleaning and Feature Engineering
Handling Missing Values:
The code examines missing values within the dataset. Depending on the amount and pattern of missingness, it uses appropriate techniques:

Imputation: Filling in missing data with mean/median/mode or using more advanced imputation methods.
Removal: Dropping rows or columns if the missing data is deemed too extensive or non-critical.
Encoding Categorical Variables:
Since machine learning models require numerical input:

The notebook applies One-Hot Encoding (using sklearn.preprocessing.OneHotEncoder) to transform categorical features into a binary format.
The code uses ColumnTransformer to selectively apply transformations to different types of columns (numeric vs. categorical).
Scaling Features:
Numerical features are standardized using StandardScaler from scikit-learn. This step is crucial for algorithms that are sensitive to the scale of input data, such as Support Vector Machines (SVM).

4. Model Building
Importing Model-Related Libraries:
A dedicated cell (e.g., cell 15) imports various machine learning models and tools:

Classification Models:
K-Nearest Neighbors (KNN) – a simple instance-based learning method.
Logistic Regression – a linear model for binary classification.
Support Vector Machine (SVM) – for finding an optimal decision boundary.
Naïve Bayes (GaussianNB, MultinomialNB, BernoulliNB) – probabilistic classifiers.
Pipeline and ColumnTransformer:
These tools are used to streamline the workflow by combining preprocessing (scaling, encoding) and model training into a single pipeline.
Hyperparameter Tuning:
The code imports GridSearchCV to perform an exhaustive search over specified parameter values for the models, ensuring that the best hyperparameters are selected based on cross-validation performance.
Evaluation Metrics:
Various evaluation functions are imported from sklearn.metrics, including functions to compute accuracy, precision, recall, F1-score, and ROC AUC.
