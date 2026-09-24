# 🏠 House Price Prediction

## 📌 Project Overview

This project focuses on predicting house prices using **Machine Learning**. The model learns from various house features such as area, number of bedrooms, bathrooms, stories, parking availability, furnishing status, and other amenities.

The project includes **data preprocessing, exploratory data analysis, feature engineering, model training, prediction, evaluation, and model comparison**.

The main objective is to build regression models that can predict house prices and compare their performance using appropriate evaluation metrics.

---

## 🗂️ Dataset

The dataset contains information about houses along with their corresponding sale prices.

### Key Features

- `price` – Sale price of the house (**Target Variable**)
- `area` – Area of the house in square feet
- `bedrooms` – Number of bedrooms
- `bathrooms` – Number of bathrooms
- `stories` – Number of stories
- `mainroad` – Whether the house is connected to the main road
- `guestroom` – Availability of a guest room
- `basement` – Availability of a basement
- `hotwaterheating` – Availability of hot water heating
- `airconditioning` – Availability of air conditioning
- `parking` – Number of parking spaces
- `prefarea` – Whether the house is located in a preferred area
- `furnishingstatus` – Furnishing status of the house

---

## 🔍 Data Exploration & Preprocessing

The following steps were performed:

- Checked the shape and structure of the dataset
- Checked for missing/null values
- Checked for duplicate records
- Analyzed numerical features
- Visualized numerical feature distributions
- Visualized house price and area distributions
- Analyzed categorical feature distributions
- Performed correlation analysis
- Studied relationships between features and house prices
- Applied standard scaling to numerical features
- Applied one-hot encoding to categorical features

---

## 📊 Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the dataset and identify relationships between different features.

### Visualizations Used

- Histograms
- Distribution plots
- Count plots
- Correlation heatmap
- Feature relationship analysis

EDA helped in understanding the distribution of house prices and identifying the relationship between different features and the target variable.

---

## 🤖 Machine Learning Models

Two regression algorithms were implemented:

### 1. Decision Tree Regressor

The Decision Tree Regressor predicts house prices by learning decision rules from the input features.

### 2. Random Forest Regressor

The Random Forest Regressor is an ensemble learning algorithm that combines multiple decision trees to produce predictions.

---

## ⚙️ Model Building Process

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Preprocessing
   ↓
One-Hot Encoding
   ↓
Feature Scaling
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Model Comparison

**Key Columns:**
- `price` — the sale price of the house  
- `area` — size of the house in square feet  
- `bedrooms`, `bathrooms`, `stories` — structural details  
- `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `parking`, `prefarea`, `furnishingstatus` — categorical features indicating amenities or house condition  

 🔍 Data Exploration & Visualization  
- Checked dataset shape, missing/null values, duplicates  
- Visualized distributions of numerical features (e.g., `price`, `area`) using histograms with KDE  
- Visualized categorical features via count plots to inspect frequency distributions  
- Performed correlation analysis and visual inspection to understand feature importance  
- Applied standard scaling to numerical features (excluding the target) and one-hot encoding for categorical features  

 🧰 Model Building  
We implemented two machine learning models to compare performance:
1. **Decision Tree Regressor**  
2. **Random Forest Regressor**

Steps included:
- Splitting data into training (80%) and testing (20%) sets  
- Training the models on the training set  
- Predicting on both training and testing sets  
- Evaluating performance using MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error)  
- Comparing model scores and selecting the more accurate one
