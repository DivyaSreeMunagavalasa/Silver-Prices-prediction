# Silver Price Prediction in India

This project aims to predict the price of silver in India using various machine learning algorithms. The data includes features such as `Open`, `High`, `Low`, `Close`, and `Volume` prices of silver. The project involves data preprocessing, data visualization, and building predictive models using Decision Trees, Random Forest Regressor, and Multiple Linear Regression.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Data Visualization](#data-visualization)
- [Model Building](#model-building)
- [Evaluation](#evaluation)
- [Contributing](#contributing)

## Overview

The goal of this project is to predict the closing price of silver based on various input features. This prediction can be useful for investors and analysts in making informed decisions about silver trading.

## Dataset

The dataset includes the following columns:

- `Date`: The date of the record.
- `Open`: The opening price of silver on a particular date.
- `High`: The highest price of silver on a particular date.
- `Low`: The lowest price of silver on a particular date.
- `Close`: The closing price of silver on a particular date.
- `Volume`: The volume of silver traded.

## Data Preprocessing

Data preprocessing steps include:

1. Loading the data using Pandas.
2. Checking for and handling missing or null values.
3. Converting data types as necessary (e.g., converting `Volume` to float).
4. Checking for duplicate values.
5. Data transformation and cleaning.

## Data Visualization

Various plots are used to explore the dataset:

- **Scatter plots** to explore relationships between `Open`, `High`, and `Close` prices.
- **Line plots** to visualize the trend in silver prices.
- **Histograms** to understand the distribution of the `Close` prices.
- **Box plots** to identify outliers in `Close` prices and `Volume`.

## Model Building

Three machine learning models are used to predict the closing price of silver:

1. **Decision Tree Regressor**: A tree-based model that splits the data into branches based on feature values.
2. **Random Forest Regressor**: An ensemble model that builds multiple decision trees and merges their results for more accurate and stable predictions.
3. **Multiple Linear Regression**: A linear approach to model the relationship between the dependent variable (`Close` price) and multiple independent variables.

The models are trained using a split of the dataset into training and testing sets.

## Evaluation

Model performance is evaluated using the R-squared score (`r2_score`), which indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.
