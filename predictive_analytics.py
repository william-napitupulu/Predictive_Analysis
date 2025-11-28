import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plot style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# 1. Data Loading
print("Loading data...")
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

# 2. Data Understanding
print("\nData Info:")
print(df.info())
print("\nData Description:")
print(df.describe())

print("\nChecking for duplicates:")
print("Jumlah duplikasi: ", df.duplicated().sum())

# Visualizations (Optional in script, but good to have code)
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.hist(df['MedHouseVal'], bins=50, edgecolor='black', alpha=0.7)
# plt.title('Distribution of House Prices')
# plt.subplot(1, 2, 2)
# plt.boxplot(df['MedHouseVal'])
# plt.title('Box Plot of House Prices')
# plt.show()

# 3. Data Preparation
print("\nSplitting data...")
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

print("\nScaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Modeling
# Random Forest
print("\nTraining Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

# Gradient Boosting
print("Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)

# Linear Regression
print("Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# SVR
print("Training SVR...")
svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_model.fit(X_train_scaled, y_train)

# 5. Evaluation
print("\nEvaluating models...")
models = {
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "Linear Regression": lr_model,
    "SVR": svr_model
}

results = []
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results.append({'Model': name, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2 Score': r2})

results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df)

best_model_row = results_df.loc[results_df['R2 Score'].idxmax()]
print(f"\nBest Model: {best_model_row['Model']} (R2 Score: {best_model_row['R2 Score']:.4f})")
