import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE_PATH = r"C:\Users\p.c\Desktop\Dataset acquition\kaggle_dataset.csv"
df = pd.read_csv(FILE_PATH)

original_counts = df['label'].value_counts()

def get_color(label):
    fall_classes = ['fall', 'lfall', 'rfall', 'light', 'ffall']
    normal_classes = ['sit', 'step', 'walk']
    
    if label in fall_classes:
        return "#d95f5f"   
    elif label in normal_classes:
        return "#5fd96b"   
    else:
        return "#7a7a7a"   

colors = [get_color(lbl) for lbl in original_counts.index]


plt.figure(figsize=(10, 6))
plt.bar(original_counts.index, original_counts.values, color=colors)

plt.title("Original Dataset Class Distribution")
plt.xlabel("Classes")
plt.ylabel("Count")
plt.xticks(rotation=45)

for i, v in enumerate(original_counts.values):
    plt.text(i, v + 5, str(v), ha='center')

plt.tight_layout()
plt.show()