# Import libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# -----------------------------
# Step 1: Create Non-Linear Data
# -----------------------------
np.random.seed(0)

X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = 3 * X**2 + 2 * X + 5 + np.random.randn(100, 1) * 5   # Quadratic relationship

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Step 2: Linear Regression (Original Feature)
# -----------------------------
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)
r2_linear = r2_score(y_test, y_pred_linear)

# -----------------------------
# Step 3: Polynomial Features (degree=2)
# -----------------------------
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

y_pred_poly = poly_model.predict(X_test_poly)
r2_poly = r2_score(y_test, y_pred_poly)

# -----------------------------
# Step 4: Print Results
# -----------------------------
print("R² Score (Linear Regression - Original Feature):", r2_linear)
print("R² Score (Linear Regression - Polynomial Features):", r2_poly)