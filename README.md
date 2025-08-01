Gold Price Prediction
This repository contains a Jupyter Notebook (Gold_Price_Prediction.ipynb) that demonstrates a machine learning model for predicting gold prices. The project involves data collection, preprocessing, analysis, and training a RandomForestRegressor model to predict the price of gold (GLD) based on related financial indicators.

Table of Contents
Overview

Dataset

Methodology

Libraries Used

Model

Results

Usage

Overview
The main objective of this project is to build a predictive model for gold prices. The notebook covers the entire machine learning pipeline, from initial data exploration to model evaluation.

Dataset
The model utilizes a dataset named gold price dataset.csv. This dataset includes the following columns:

Date: The date of the observation.

SPX: S&P 500 index.

GLD: Gold price (target variable).

USO: United States Oil Fund.

SLV: Silver price.

EUR/USD: Euro to US Dollar exchange rate.

The dataset contains 2290 rows and 6 columns, with no missing values.

Methodology
The project follows these key steps:

Importing Libraries: Essential Python libraries for data manipulation, visualization, and machine learning are imported.

Data Collection and Processing:

The gold price dataset.csv is loaded into a Pandas DataFrame.

Initial data inspection is performed using head(), tail(), shape, info(), and isnull().sum() to understand the data structure, check for missing values, and get basic information.

Statistical measures of the data are obtained using describe().

Correlation Analysis:

The correlation between features is calculated.

A heatmap is constructed using seaborn to visualize the correlation matrix, helping to understand positive and negative correlations between variables.

Data Pre-processing:

The "Date" column is converted to datetime objects.

Feature selection is performed to prepare the data for the model.

Data Analysis and Visualization:

Distribution of the GLD Price is visualized.

Splitting Data: The dataset is split into training and test sets.

Model Training: A RandomForestRegressor model is trained on the preprocessed training data.

Prediction on Test Data: The trained model makes predictions on the unseen test set.

Model Evaluation:

The R-squared error is calculated to evaluate the model's performance.

Visualizing Actual vs. Predicted Prices: A plot is generated to compare the actual gold prices with the predicted gold prices, providing a visual assessment of the model's accuracy.

Libraries Used
numpy

pandas

matplotlib.pyplot

seaborn

sklearn.model_selection.train_test_split

sklearn.ensemble.RandomForestRegressor

sklearn.metrics

Model
The machine learning model used for gold price prediction is RandomForestRegressor.

Results
The correlation analysis shows the relationships between GLD and other financial instruments like SPX, USO, and SLV, as well as the EUR/USD exchange rate.

The model's performance is evaluated using the R-squared error, providing an indication of how well the model predicts new data.
