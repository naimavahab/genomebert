import pandas as pd

f = 'mortality_data.csv'
df = pd.read_csv(f)

df['length'] = df['feature'].apply(len)

# Find the maximum and minimum length
max_length = df['length'].max()
min_length = df['length'].min()

print(f"Maximum length: {max_length}")
print(f"Minimum length: {min_length}")

count_zero_label = (df['labels'] == 0).sum()

print(f"Number of rows with label as 0: {count_zero_label}")

count_zero_label = (df['labels'] == 1).sum()

print(f"Number of rows with label as 1: {count_zero_label}")

