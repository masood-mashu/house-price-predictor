import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target

# Features & target
X = df.drop('Price', axis=1)
y = df['Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features (needed for linear models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models dict — tree models use raw data, linear use scaled
models = {
    'Linear Regression': (LinearRegression(),       True),
    'Ridge':             (Ridge(alpha=1.0),          True),
    'Lasso':             (Lasso(alpha=0.1),          True),
    'Random Forest':     (RandomForestRegressor(n_estimators=100, random_state=42), False),
    'Gradient Boosting': (GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42), False),
}

print("=" * 55)
print("           MODEL COMPARISON")
print("=" * 55)

results = {}
for name, (model, use_scaled) in models.items():
    X_tr = X_train_scaled if use_scaled else X_train
    X_te = X_test_scaled  if use_scaled else X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    results[name] = {'model': model, 'rmse': rmse, 'r2': r2, 'scaled': use_scaled}

    print(f"\n{name}:")
    print(f"  RMSE : {rmse:.4f}  (${rmse*100000:,.0f})")
    print(f"  R²   : {r2:.4f}  ({round(r2*100, 1)}% variance explained)")

print("\n" + "=" * 55)
best = min(results, key=lambda x: results[x]['rmse'])
print(f"✅ Best Model : {best}")
print(f"   RMSE       : ${results[best]['rmse']*100000:,.0f}")
print(f"   R²         : {results[best]['r2']*100:.1f}%")
print("=" * 55)