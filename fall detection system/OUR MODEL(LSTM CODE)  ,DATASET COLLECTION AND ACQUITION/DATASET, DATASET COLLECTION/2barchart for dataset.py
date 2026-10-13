import pandas as pd
import matplotlib.pyplot as plt

FILE_PATH = r"C:\Users\p.c\Desktop\Dataset acquition\kaggle_dataset.csv"
df = pd.read_csv(FILE_PATH)

def label_map(act):
    if act in ['fall', 'lfall', 'rfall', 'light', 'ffall']:
        return "Fall"
    elif act in ['sit', 'step', 'walk']:
        return "Normal"
    else:
        return None

df['mapped_label'] = df['label'].apply(label_map)
df = df.dropna(subset=['mapped_label'])

counts = df['mapped_label'].value_counts()

colors = {
    "Fall": "#d95f5f",     
    "Normal": "#5fd96b"    
}

bar_colors = [colors[label] for label in counts.index]


plt.figure(figsize=(6, 5))
plt.bar(counts.index, counts.values, color=bar_colors)

plt.title("Fall vs Normal Activity Distribution")
plt.xlabel("Activity Type")
plt.ylabel("Count")

# Show values on bars
for i, v in enumerate(counts.values):
    plt.text(i, v + 5, str(v), ha='center')

plt.tight_layout()
plt.show()