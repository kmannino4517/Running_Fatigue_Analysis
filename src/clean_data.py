import pandas as pd

df = pd.read_csv("../data/raw/runs.csv")

print(df.head())
print(df.describe())