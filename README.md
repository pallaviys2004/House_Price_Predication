House Price Prediction Project

📌 Project Overview  
This project aims to build a predictive model for house prices based on a given dataset. Using data preprocessing, feature engineering, and machine learning algorithms, we explore and predict house sale prices with the goal of minimizing prediction error and understanding key drivers of housing price variation.

🗂 Dataset  
The dataset was obtained from a public source (e.g., [Kaggle](https://www.kaggle.com/search?q=house+price+prediction+dataset)). It includes various features such as area, number of bedrooms, bathrooms, stories, parking availability, furnishing status, and more, along with the target variable `price`.

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