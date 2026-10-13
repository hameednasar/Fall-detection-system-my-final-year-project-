import pandas as pd
import matplotlib.pyplot as plt

FILE_PATH = r"C:\Users\p.c\Desktop\Dataset acquition\kaggle_dataset.csv"

df = pd.read_csv(FILE_PATH)

original_counts = df['label'].value_counts()

plt.figure(figsize=(7, 7))
plt.pie(
    original_counts.values,
    labels=original_counts.index,
    autopct='%1.1f%%',
    startangle=90
)
plt.title("Original Dataset Class Distribution")
plt.axis("equal")
plt.show()


def label_map(act):
    if act in ['fall', 'lfall', 'rfall', 'light', 'ffall']:
        return "Fall (1)"
    elif act in ['sit', 'step', 'walk']:
        return "Normal (0)"
    else:
        return "Unknown"

df['mapped_label'] = df['label'].apply(label_map)
df = df[df['mapped_label'] != "Unknown"]

mapped_counts = df['mapped_label'].value_counts()

plt.figure(figsize=(7, 7))
plt.pie(
    mapped_counts.values,
    labels=mapped_counts.index,
    autopct='%1.1f%%',
    startangle=90
)
plt.title("Binary Class Distribution (After Mapping)")
plt.axis("equal")
plt.show()