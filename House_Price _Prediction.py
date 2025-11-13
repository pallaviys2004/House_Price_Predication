# Importing necessary libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For enhanced data visualization
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import RandomForestRegressor  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import mean_absolute_error, mean_squared_error  
# Reading the dataset
df = pd.read_csv('Housing (3).csv')  
# Displaying the first 10 rows of the dataset
df.head(10)
# Checking basic information about the dataset (e.g., column names, data types, and non-null counts)
df.info()
# Checking for missing values in the dataset
df.isna().sum()
# Checking for duplicate rows in the dataset
df.duplicated().sum()

# Displaying the column names
df.columns
# Checking the shape of the dataset (number of rows and columns)
df.shape
# Separating numerical and categorical columns
num_col = ['price', 'area']  # Numerical columns
cat_col = [
    'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement',
    'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus'
]  # Categorical columns
# Plotting histograms for numerical columns
for i in num_col:
    print(f'Histogram of {i}')
    sns.histplot(data=df, x=i, kde=True)
    plt.title(f'Histogram of {i}')
    plt.show()
# Plotting bar plots for categorical columns
for i in cat_col:
    sns.countplot(data=df, x=i)
    plt.title(f'Bar plot of {i}')
    plt.show()
# Scaling the numerical columns (excluding 'price' as it is the target variable)
scaler = StandardScaler()
scaled_num_col = scaler.fit_transform(df[num_col])  # Scaling numerical columns
scaled_num_col = pd.DataFrame(scaled_num_col, columns=num_col)
scaled_num_col = scaled_num_col.drop('price', axis=1)  # Dropping 'price' from scaled data
scaled_num_col.head()
# Encoding categorical columns into dummy/one-hot variables
cat_col_with_dummies = pd.get_dummies(data=df[cat_col], columns=cat_col, drop_first=True)
cat_col_with_dummies.head()
# Combining scaled numerical columns and encoded categorical columns into one dataframe
new_df = pd.concat([scaled_num_col, cat_col_with_dummies], axis=1)
new_df.head()
# Separating features (X) and target variable (y)
X = new_df
y = df.price
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# Initializing models: Decision Tree and Random Forest
dt = DecisionTreeRegressor()
rf = RandomForestRegressor()
models = [dt, rf]
# Training and evaluating models
for model in models:
    print(f'For {model} : ')
    # Fitting the model to training data
    model.fit(X_train, y_train)
    

    # Predicting on training and testing datasets
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    # Evaluating performance on training dataset
    print('For training dataset:')
    mae_train = mean_absolute_error(y_train, y_pred_train)
    print('MAE train:', mae_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    print('RMSE train:', np.sqrt(mse_train))
    # Evaluating performance on testing dataset
    print('For testing dataset:')
    mae_test = mean_absolute_error(y_test, y_pred_test)
    print('MAE test:', mae_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    print('RMSE test:', np.sqrt(mse_test))   
    # Printing model score
    print('Score:', model.score(X_train, y_train))
    print('*' * 80)
    print()



