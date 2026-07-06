# House Price Prediction Project
this project involves building a ML model to predict house price base on various features. the dataset used for this project is from kaggle "Housing.csv".
the goal is to develop a model that accurately predicts house prices given a set of input features

# File Structure
    Notebook in the form of .ipynb form 
    data in the form of .csv 
    file contain a .pkl model for regression model

# Linraries Used
Numpy 
pandas 
scikit-learn
Ridge
Lasso

# Data Loading and Analysis
The training and test datasets are loaded from CSV files.
Exploratory data analysis is performed to understand the structure and characteristics of the data.

# Data Preprocessing
Missing values are handled using appropriate techniques such as imputation or dropping columns.
Categorical variables are encoded using one-hot encoding.
Numerical features are standardized to ensure uniformity and improve model performance.

# Model Selection and Training

Several regression models are considered, including Linear Regression, SGDRegressor, DecisionTreeRegressor, RandomForestRegressor, XGBRegressor.
Cross-validation is used to evaluate each model's performance based on the R-squared score.
The GradientBoostingRegressor model is selected based on its superior performance.

# Model Evaluation and Prediction
The selected model is trained on the training dataset.
The trained model is used to make predictions on the test dataset.
The predictions are saved to a CSV file (submission.csv) for submission.