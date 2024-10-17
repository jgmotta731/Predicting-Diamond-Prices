# Predicting Diamond Prices

**Author**: Jack Motta  
**Objective**: To build an accurate predictive price model for diamonds and compare the performance of Boosting, Random Forest, and Penalized Regression models. The goal is to prioritize prediction over inference.

---

## Table of Contents

1. [Overview](#overview)
2. [Attributes](#attributes)
3. [Objective](#objective)
4. [Methodology](#methodology)
5. [Feature Engineering](#feature-engineering)
6. [Preprocessing](#preprocessing)
    - [Random Forest and XGBoost](#preprocessing-steps-rf-and-xgb)
    - [Penalized Regression](#preprocessing-steps-penalized-regression)
7. [Model Tuning](#model-tuning)
8. [Results](#results)
9. [Visualizations](#visualizations)
10. [Libraries](#libraries)

---

## Overview

This project analyzes a dataset of diamonds, containing 53,940 rows and 11 variables. The primary objective is to predict the **price** of diamonds using various machine learning techniques. The data contains 9 predictors, and we will perform feature engineering and data preprocessing to enhance model performance.

---

## Attributes

- **53,940 diamonds** with 11 variables.
- **Predictors**:
    - Carat
    - Cut
    - Color (graded from "J" (worst) to "D" (best))
    - Clarity (graded from worst to best: I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF)
    - Dimensions: Length (x), Width (y), Depth (z)
    - Total depth percentage, calculated as: `2 * z / (x + y)`
    - Table: Width of the top of the diamond relative to the widest point.
- **Outcome**: Price

---

## Objective

- To build an accurate model to predict the price of diamonds.
- Compare the performance of:
    - **Random Forest**
    - **XGBoost**
    - **Penalized Regression (Elastic Net)**

---

## Methodology

- Perform data preprocessing, feature engineering, and exploratory data analysis (EDA).
- Apply a 70/30 train-test split.
- Use **RMSE** as the primary performance metric and **R-squared** and **MSE** as secondary metrics.
- Feature importance will be evaluated using **Interpretable Machine Learning (IML)**.

---

## Feature Engineering

- **Volume**: Approximate volume using `length * width * depth`.
- **Aspect Ratio**: `length/width` to approximate the diamond's shape.
- **Surface Area**: Modeled as an ellipsoid, surface area approximates how light interacts with the diamond, impacting its desirability and price.
- Shape categorization based on aspect ratio.
- **ID Renaming**: Renamed "X" to "ID".

---

## Preprocessing

### Preprocessing Steps (RF and XGBoost)

- Impute missing values for `x`, `y`, and `z` using KNN due to likely data entry errors.
- Normalize all numeric predictors.
- One-hot encode nominal predictors.
- Use feature-engineered variables (volume, shape, surface area) in the models.

### Preprocessing Steps (Penalized Regression)

- KNN imputation for missing values.
- Remove redundant predictors (`x`, `y`, `z`, aspect ratio).
- Log-transform price to handle skewness.
- Remove near-zero variance predictors and correlated predictors (threshold = 0.8).
- Fit cubic splines on **volume** with 3 degrees of freedom for flexibility in capturing non-linearity.

---

## Model Tuning

- **Random Forest**: Tune the minimum node size and randomly sample predictors (mtry = 3) with 500 trees.
- **XGBoost**: Tune parameters such as max depth, learning rate, loss reduction, subsample size, and boosting iterations.
- **Penalized Regression (Elastic Net)**: Tune both penalty and mixture parameters for optimal performance.

---

## Results

### Model Performance Metrics

| Model          | RMSE  | MAE   | R-Squared |
|----------------|-------|-------|-----------|
| Random Forest  |  990.399   | 609.9204    | 0.956        |
| XGBoost        |  555.513   | 287.3736    | 0.981        |
| Elastic Net    |  20001.305   | 904.5414    | 0.751        |

---

## Visualizations

The project includes the following visualizations:

- **Partial Dependence Plots (PDP)** for key predictors like `carat`, `volume`, and `surface_area`.
- **Variable Importance Plots** for Random Forest, XGBoost, and Elastic Net models.
- **ALE Plots** to further interpret the relationships between predictors and price.

---

## Libraries

The following R libraries were used in this project:

```r
library(tidymodels)
library(tidyverse)
library(dplyr)
library(vip)
library(iml)
library(ggplot2)
library(xgboost)
library(ranger)
library(ggthemes)
library(doParallel)
library(themis)
library(future)
library(finetune)
library(probably)
library(pdp)
library(yardstick)
library(gtsummary)
library(gridExtra)
library(dials)
library(rpart)
library(rpart.plot)
