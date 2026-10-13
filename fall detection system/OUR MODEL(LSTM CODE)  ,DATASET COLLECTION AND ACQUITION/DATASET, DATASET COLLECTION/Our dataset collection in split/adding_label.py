import pandas as pd

file_path = r"C:\Users\p.c\Desktop\Dataset acquition\walk.csv"

label_name = "walk"  
df = pd.read_csv(file_path, header=0)

df.columns = ["xAcc", "yAcc", "zAcc", "xGyro", "yGyro", "zGyro"]

df["label"] = label_name

df.to_csv(file_path, index=False)

print("Header fixed and label column added!")