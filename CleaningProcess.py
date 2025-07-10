# Data manipulation and numerical computing
import pandas as pd
import numpy as np

# Machine learning utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical modeling
import statsmodels.api as sm
from statsmodels.formula.api import ols

# System utilities
import pickle
import os
import warnings

# Machine learning pipeline utilities
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR

# Gradient boosting libraries
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


# Load Dataset
def load_data(file_path):
    df = pd.read_excel(file_path)
    return df

file_path = 'raw_data.xlsx'
df = load_data(file_path)

# Split test data for analysis
X = df.drop("Price", axis = 1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

df = X_train.copy()
df['Price']= y_train

# Handle missing value
print("--- Handle Missing Value below ---")
for col in df.columns:
    number_null = df.loc[: , col].isnull().sum()
    perc_null = (number_null / df.shape[0]) * 100
    print ('{} - {} - %{}'.format(col,number_null, round(perc_null,3)))

# Numerical Sanity Check 
print("--- Numberical Sanity Check below ---")
numeric_columns = df.select_dtypes(include=['int64','float64']).columns.tolist()
print(df[numeric_columns].describe())

# Categorical Sanity Check
print("--- Categorical Sanity Check below ---")
cat_columns = df.select_dtypes(exclude=['int64','float64']).columns.tolist()

def Cat_Checker(CatIndex):
    values = np.sort(df[cat_columns[CatIndex]].unique())
    return cat_columns[CatIndex], values

print(Cat_Checker(0))
print(Cat_Checker(1))
print(Cat_Checker(2))
print(Cat_Checker(3))

# Correlation Analysis (Pearson correlation)
fig, axes = plt.subplots(2,3,figsize=(25,8))
axes = axes.ravel()

for i, col in enumerate(numeric_columns):
    sns.regplot(x=col,
                y='Price',
                data=df,
                ax=axes[i],
                scatter_kws={"color": "red"},
                line_kws={"color": "black"})
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('')
plt.tight_layout()
plt.show()

# Outlier Detection

def box_plot(columns):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18,6))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f'{col}')

    plt.tight_layout()
    plt.show()

# Example 
box_plot(numeric_columns[:6])

# Categorical Encoding 
# Mean-Target Encoding

# Brand
brand_mean_price = df.groupby("Brand")['Price'].mean()
df['Encoded_Brand'] = df['Brand'].map(brand_mean_price)

# Model
model_mean_price = df.groupby('Model')['Price'].mean()
df['Encoded_Model'] = df['Model'].map(model_mean_price)

#Drop
df.drop(['Brand','Model'], axis=1, inplace=True)



# Ensure models directory exists
models_dir = 'models/'
os.makedirs(models_dir, exist_ok=True)


with open('models/Brand_Encoder.pkl','wb') as f:
    pickle.dump(brand_mean_price, f)

with open('models/Model_Encoder.pkl','wb') as f:
    pickle.dump(model_mean_price, f)

# One-Hot-Encoding

categorical_cols = ['Fuel', 'Transmission']

# Define OneHotEncoder
encoder = OneHotEncoder(drop='first',sparse_output=False, handle_unknown='ignore')

#Fit and Transform only selected categorical columns 
encoded_array = encoder.fit_transform(df[categorical_cols])

#Convert to DataFrame
encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(categorical_cols))

# Drop original categorical columns and merge encoded ones
df_encoded = df.drop(columns=categorical_cols).reset_index(drop=True)
df = pd.concat([df_encoded, encoded_df], axis=1)

with open('models/OneHot_Encoder.pkl','wb') as f:
    pickle.dump(encoder, f)

# Data Splitting 
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size = 0.2, random_state = 42)

# Modelling

def evaluate_model(X_train, y_train, X_val, y_val, model):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cross_val_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    cross_val_rmse = -cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')

    print(f"Training CV R2: {cross_val_r2.mean():.4f}, Training CV RMSE: {cross_val_rmse.mean():.4f}")

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    r2_train = r2_score(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    r2_val = r2_score(y_val, y_val_pred)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    
    print(f"Training R2: {r2_train:.4f}, Training RMSE: {rmse_train:.4f}")
    print(f"Validation R2: {r2_val:.4f}, Validation RMSE: {rmse_val:.4f}")

# Model Training
# Linear Regression 

print("---- Model Training below ----")
print("1. Linear Regression")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("lin_reg", LinearRegression())
])
print(evaluate_model(X_train, y_train, X_val, y_val, model))

print("2. Decision Tree")
model = Pipeline([
    ("DecisionTree_reg", DecisionTreeRegressor())
])
print(evaluate_model(X_train, y_train, X_val, y_val, model))

print("3. Random Forest Regressor")
model = Pipeline([
    ("RandomForest_reg", RandomForestRegressor())
])
print(evaluate_model(X_train, y_train, X_val, y_val, model))

print("4. XGBoost Regressor")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBRegressor(objective='reg:squarederror'))
])
print(evaluate_model(X_train, y_train, X_val, y_val, model))

print("5. CatBoost Regression")
model = Pipeline([
    ("catboost", CatBoostRegressor())
])
print(evaluate_model(X_train, y_train, X_val, y_val, model))

# Looking at the result, we can come to the conclusion that CatBoost is the best performing model

final_model = model


# Apply model on Test Data
# We can predict the price on the test data

with open('models/Brand_Encoder.pkl','rb') as f:
    brand_encoder = pickle.load(f)

with open('models/Model_Encoder.pkl','rb') as f:
    model_encoder = pickle.load(f)

with open('models/OneHot_Encoder.pkl','rb') as f:
    onehot_encoder = pickle.load(f)

# Next, encode categorical features 
X_test['Encoded_Brand'] = X_test['Brand'].map(brand_encoder)
X_test['Encoded_Model'] = X_test['Model'].map(model_encoder)
X_test['Encoded_Brand'].fillna(X_test['Encoded_Brand'].mean(), inplace=True)
X_test['Encoded_Model'].fillna(X_test['Encoded_Brand'].mean(), inplace=True)
X_test.drop(['Brand','Model'], axis=1, inplace=True)

#Encoding Fuel and Transmission
categorical_cols = ['Fuel' , 'Transmission']
encoded_array_test = onehot_encoder.transform(X_test[categorical_cols])
encoded_df_test = pd.DataFrame(encoded_array_test, columns = onehot_encoder.get_feature_names_out(categorical_cols))

#Merge encoded Columns with test data 
X_test_encoded = X_test.drop(columns = categorical_cols).reset_index(drop=True)
X_test = pd.concat([X_test_encoded, encoded_df_test], axis=1)

# Perdict the test data
y_test_pred = final_model.predict(X_test)

r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("--- Predicting the Test Data ---")
print(f"Test R2: {r2_test:.4f}, Test RMSE: {rmse_test:.4f}")

pickle.dump(final_model, open('models/Model.pkl','wb'))