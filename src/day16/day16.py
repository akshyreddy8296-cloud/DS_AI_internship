import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

normal_data = np.random.normal(170, 10, 500)
right_skewed = np.random.exponential(2, 500) 
left_skewed = 100 - np.random.exponential(2, 500)

datasets = {
    "Normal": normal_data,
    "Right-Skewed": right_skewed,
    "Left-Skewed": left_skewed
}

for name, data in datasets.items():
    df = pd.DataFrame({"Value": data})

    mean = df["Value"].mean()
    median = df["Value"].median()

    plt.figure() 
    sns.histplot(df["Value"], kde=True)
    plt.title(f"{name} | Mean={mean:.2f}, Median={median:.2f}")
    plt.show()

    print(f"{name} → Mean: {mean:.2f}, Median:{median:.2f}")
    
    
    
 #### task 2 #####
import numpy as np
import pandas as pd

# 1️⃣ Generate Sample Dataset (Normal distribution with some extreme values)
np.random.seed(42)

data = np.random.normal(loc=50, scale=10, size=1000)

# Add some extreme outliers manually
data = np.append(data, [150, -20, 200])

# Create DataFrame
df = pd.DataFrame(data, columns=["value"])

# 2️⃣ Calculate Mean (μ) and Standard Deviation (σ)
mu = df["value"].mean()
sigma = df["value"].std()

print("Mean (μ):", mu)
print("Standard Deviation (σ):", sigma)

# 3️⃣ Calculate Z-Score
df["z_score"] = (df["value"] - mu) / sigma

# 4️⃣ Identify Outliers (|Z| > 3)
outliers = df[np.abs(df["z_score"]) > 3]

print("\nStatistical Outliers (|Z| > 3):")
print(outliers)

print("\nTotal Outliers Found:", len(outliers))







###### task 3 #####
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

population = np.random.exponential(scale=50000, size=100000)

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
sns.histplot(population, kde=True)
plt.title("Original Population (Skewed Income Data)")

sample_means = []

for i in range(1000):
    sample = np.random.choice(population, size=30)
    sample_mean = np.mean(sample)
    sample_means.append(sample_mean)

plt.subplot(1,2,2)
sns.histplot(sample_means, kde=True)
plt.title("Distribution of 1000 Sample Means (n=30)")

plt.tight_layout()
plt.show()

print("Population Mean:", np.mean(population))
print("Mean of Sample Means:", np.mean(sample_means))





