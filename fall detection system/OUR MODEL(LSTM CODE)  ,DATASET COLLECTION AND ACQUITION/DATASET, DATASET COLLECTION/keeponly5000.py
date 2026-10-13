import pandas as pd

file_path = r"C:\Users\p.c\Desktop\Dataset acquition\lfall.csv"

df = pd.read_csv(file_path)

df_trimmed = df.head(5000)

df_trimmed.to_csv(file_path, index=False)

print("File trimmed to 5000 rows (5001 lines with header) and overwritten!")