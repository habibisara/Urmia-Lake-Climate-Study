# === Import Required Libraries ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import shap

# === Step 1: Load and Clean the Data ===
df = pd.read_csv("E:/Nava/Urmia Lake Research/Urmia_Lake_Climate_Data_month.csv")
df = df.drop(columns=['Unnamed: 7'], errors='ignore')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')
df.set_index('date', inplace=True)
df = df.interpolate(method='linear')

# === Step 2: Remove wind and create lag/seasonal features ===
# Lag features
df['LWE_t-1'] = df['LWE'].shift(1)
df['LWE_t-2'] = df['LWE'].shift(2)
df['LWE_t-3'] = df['LWE'].shift(3)
df['LWE_avg_3'] = df['LWE'].rolling(3).mean().shift(1)
df['LWE_diff'] = df['LWE'].diff().shift(1)

# Cyclical month encoding
df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

# Drop NA values introduced by shifting/rolling
df = df.dropna()

# === Step 3: Define Features and Target ===
X = df.drop(columns=['LWE'])
y = df['LWE']

# === Step 4: Scale the Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Step 5: Train-Test Split (last 24 months = test) ===
train_size = len(df) - 24
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
test_dates = df.index[-24:]

# === Step 6: XGBoost Feature Importance ===
xgb_model = xgb.XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
feature_importance = xgb_model.feature_importances_

# Create SHAP explainer and compute values
explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_test)

# Summary plot of SHAP values
shap.summary_plot(shap_values, X_test, feature_names=X.columns)

# Create importance DataFrame
feature_names = X.columns
importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
importance_df = importance_df.sort_values(by='importance', ascending=False)

# === Step 7: Apply Feature Weighting ===
X_weighted_train = X_train * feature_importance
X_weighted_test = X_test * feature_importance

# === Step 8: Train RidgeCV with 5-Fold Cross-Validation ===
ridge_model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)
ridge_model.fit(X_weighted_train, y_train)
y_pred_ridge = ridge_model.predict(X_weighted_test)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)

# === Step 9: Train Random Forest with RandomizedSearchCV (5-Fold Cross-Validation) ===
param_dist = {
    'n_estimators': [50, 100, 150],
    'max_depth': [4, 6, 8, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring='r2',
    random_state=42,
    n_jobs=-1
)

rf_random_search.fit(X_weighted_train, y_train)
best_rf_model = rf_random_search.best_estimator_

# Predict with the best RF model
y_pred_rf = best_rf_model.predict(X_weighted_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

# === Step 10: Plot Feature Importance ===
plt.figure(figsize=(8, 5))
sns.barplot(data=importance_df, x='importance', y='feature', palette='coolwarm')
plt.title("\U0001F33F XGBoost Feature Importance for LWE", fontsize=14)
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# === Step 11: Plot Actual vs Predicted ===
plt.figure(figsize=(10, 5))
plt.plot(test_dates, y_test.values, label='Actual LWE', marker='o')
plt.plot(test_dates, y_pred_ridge, label='Ridge Prediction (Weighted)', marker='x')
plt.plot(test_dates, y_pred_rf, label='Random Forest Prediction (Weighted)', marker='s')
plt.title("\U0001F4C8 Actual vs Predicted LWE (Ridge vs RF)", fontsize=14)
plt.xlabel("Date")
plt.ylabel("LWE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Step 12: Print Final Performance ===
print("✅ Ridge Regression (Weighted):")
print(f"RMSE: {rmse_ridge:.2f}")
print(f"R² Score: {r2_ridge:.2f}\n")

print("\U0001F333 Random Forest (Weighted and Cross-Validated):")
print(f"RMSE: {rmse_rf:.2f}")
print(f"R² Score: {r2_rf:.2f}")