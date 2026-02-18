# ==========================================================
# TASK-1(The Distribution Deep-Dive (Univariate Analysis))
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("housing_prices.csv")

sns.set(style = "whitegrid")
plt.figure(figsize=(8,5))
sns.histplot(df["Price"], kde=True)
plt.title("Distribution of Housing Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()


skewness = df["Price"].skew()
kurtosis = df["Price"].kurt()

print("Skewness of Price:", skewness)
print("Kurtosis of Price:", kurtosis)

plt.figure(figsize=(8,5))
sns.countplot(x=df["City"])
plt.title("Count of Houses by City")
plt.xticks(rotation=45)
plt.show()


# ==========================================================
# TASK-2(The Relationship Map (Bivariate Analysis))
# =========================================================


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("housing_size_price.csv")

sns.set(style = "whitegrid")
plt.figure(figsize=(8,5))
sns.scatterplot(x=df["SquareFoot"], y=df["Price"])
plt.title("SquareFoot vs Price")
plt.xlabel("Square Foot")
plt.ylabel("Price")
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x=df["LocationType"], y=df["Price"])
plt.title("LocationType vs Price")
plt.xticks(rotation=45)
plt.show()

print("\n-------SAMPLE INSIGHTS-------")
print("1. Price increases with squareFoot & LocationType(positive correlation).")
print("2. LocationType Urban shows higher Price range.")
print("3. LocationType Rural shows low Price range.")
print("4. No extreme outliers detected.")


# ==========================================================
# TASK-3(The Pattern Finder (Correlation & Outliers))
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('housing.csv')
sns.set(style='whitegrid')

correlation = df.corr(numeric_only=True)
sns.heatmap(correlation, annot =True, cmap='coolwarm')
plt.show()

sns.boxplot(df['Price'])
plt.show()

corr_matrix = df.corr(numeric_only=True)
print("\nCorrelation Matrix:")
print(corr_matrix)
print("\nstatistical summaries.")
print(df.describe())














