import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Create sample dataset
data = {
    "Transmission": ["Automatic", "Manual", "Automatic", "Manual"],
    "Color": ["Red", "Blue", "Green", "Red"]
}

df = pd.DataFrame(data)

print("Original Data:\n")
print(df)


# Step 2: Label Encoding for Transmission (Binary)
le = LabelEncoder()
df["Transmission"] = le.fit_transform(df["Transmission"])


# Step 3: One-Hot Encoding for Color (Nominal)
df = pd.get_dummies(df, columns=["Color"], drop_first=True)

print("\nEncoded Data:\n")
print(df)
