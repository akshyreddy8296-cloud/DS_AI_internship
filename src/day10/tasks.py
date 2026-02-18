
print("Task 1: The Integrity Audit (Missing Values & Duplicates)")
import pandas as pd
df = pd.read_csv("customer_orders.csv")
pd.set_option("display.max_columns", None)
print("missing_val:\n",df.isna().sum())
print(df)
df["Name"] = df["Name"].fillna("Unknown")
df["City"] = df["City"].fillna("Unknown")
df["OrderAmount"] = df["OrderAmount"].fillna(df["OrderAmount"].mean())
df["PaymentMethod"] = df["PaymentMethod"].fillna(df["PaymentMethod"].mode()[0])
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())
removed_duplicates = df.drop_duplicates()
print("removed_duplicates:",removed_duplicates)
print(f'shape of the DataFrame after cleaning : {removed_duplicates.shape}')






































print("Task 2: The Type Fixer (Data Type Conversion)")
purchases = {
    "Price" : ["$120.50","$89.99","$250.00","$45.75","$120.50"],
    "Date" : ['2024-01-05','2024-01-10','2024-02-01','2024-02-15','2024-02-15'],
    "Product" : ["Phone","Headphones","Laptop","Mouse","Phone"]
}
df = pd.DataFrame(purchases)
print(df.to_csv('sales.csv'))
print(df.dtypes,"\n")
Price_data= df['Price'].str.replace('$',"")
print(Price_data.astype(float),"\n")
df['Date'] = pd.to_datetime(df['Date'])
print(df['Date'],"\n")
print(df)















print("Task 3: The Categorical Standardizer (String Cleaning)")
details = {
    "Location": ["new York", "new york",  "new york","chicago", "chicago","CHICAGO"],
    "Sales": [100, 150, 200, 120,300,400]
}
df= pd.DataFrame(details)
location = df['Location']
print(location.str.strip())
print(location.str.lower())

data = {
    "Location": [" New York", "new york", "NEW YORK ", "New York"]
}

df = pd.DataFrame(data)

print("Before normalization:", df["Location"].unique())
print("After normalization:", df["Location"].unique())



 









 






























