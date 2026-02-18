import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
data = {
    "Age": [22, 25, 30, 35, 40, 28, 32, 45, 50, 29],
    "Salary": [25000, 30000, 50000, 70000, 90000, 45000, 60000, 120000, 150000, 52000]
}

df = pd.DataFrame(data)

standard_scaler = StandardScaler()
df_standardized = pd.DataFrame(
    standard_scaler.fit_transform(df),
    columns=df.columns
)

minmax_scaler = MinMaxScaler()
df_normalized = pd.DataFrame(
    minmax_scaler.fit_transform(df),
    columns=df.columns
)

plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.hist(df["Salary"], bins=5)
plt.title("Original Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
 
plt.subplot(1,3,2)
plt.hist(df_standardized["Salary"], bins=5)
plt.title("Standardized Salary (Mean=0, Std=1)")
plt.xlabel("Standardized Salary")
plt.ylabel("Frequency")

plt.subplot(1,3,3)
plt.hist(df_normalized["Salary"], bins=5)
plt.title("Normalized Salary (0 to 1 Range)")
plt.xlabel("Normalized Salary")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
data = {
    "Age": [22, 25, 30, 35, 40, 28, 32, 45, 50, 29],
    "Salary": [25000, 30000, 50000, 70000, 90000, 45000, 60000, 120000, 150000, 52000]
}

df = pd.DataFrame(data)

standard_scaler = StandardScaler()
df_standardized = pd.DataFrame(
    standard_scaler.fit_transform(df),
    columns=df.columns
)

minmax_scaler = MinMaxScaler()
df_normalized = pd.DataFrame(
    minmax_scaler.fit_transform(df),
    columns=df.columns
)

plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.hist(df["Salary"], bins=5)
plt.title("Original Salary Distribution")
plt.xlabel("Salary")
plt.ylabel("Frequency")
 
plt.subplot(1,3,2)
plt.hist(df_standardized["Salary"], bins=5)
plt.title("Standardized Salary (Mean=0, Std=1)")
plt.xlabel("Standardized Salary")
plt.ylabel("Frequency")

plt.subplot(1,3,3)
plt.hist(df_normalized["Salary"], bins=5)
plt.title("Normalized Salary (0 to 1 Range)")
plt.xlabel("Normalized Salary")
plt.ylabel("Frequency")

plt.tight_layot()
plt.show()