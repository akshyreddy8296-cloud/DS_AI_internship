#write in the file
file=open("text_file_demo.py","w")
file.write("Hello, this is a file handling in python.")
file.close()

# read in the file
file=open("text_file_demo.py","r")
print(file.read())
file.close()

with open("text_file_demo.py","r")as file:
    content=file.read()
    print(content)

try:
    with open("missing_file.txt","r")as file:
        print(file.read())
except FileNotFoundError:
    print("file is missing.")
    
import csv
with open("data.csv","w",newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Age", "City"])
    writer.writerow(["akhil", 30, "New York"])
    writer.writerow(["kumar", 25, "Los Angeles"])
    writer.writerow(["sharath", 28, "Chicago"])
    writer.writerow(["akash"])
with open("data.csv", "r") as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)

import pandas as pd

df=pd.read_excel(r"C:\Users\kumar\DS_AI_Internship\src\day07\data.xlsx")
print(df)
print(df[["Name"]])
print(df.head())      # first 5 rows
print(df.tail())      # last 5 rows
print(df.shape)       # (rows, columns)
print(df.columns)     # column names
print(df.info())      # data types + non-null count
print(df.describe())  # statistics for numeric columns

        

            