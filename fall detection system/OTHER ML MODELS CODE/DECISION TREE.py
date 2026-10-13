
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


FILE_PATH   = r"C:\Users\p.c\Desktop\Data collection\acc_gyr.csv"
WINDOW_SIZE = 40
STEP_SIZE   = 5
FEATURES    = ['xAcc', 'yAcc', 'zAcc', 'xGyro', 'yGyro', 'zGyro']
N_FEATURES  = 6


def prepare_data():
    X_list, y_list = [], []
    if not os.path.exists(FILE_PATH):
        print(" File not found:", FILE_PATH)
        return np.array([]), np.array([])

    df = pd.read_csv(FILE_PATH)
    df['accel_mag'] = np.sqrt(df['xAcc']**2 + df['yAcc']**2 + df['zAcc']**2)

    def label_map(act):
        if act in ['fall', 'lfall', 'rfall', 'light']: return 1
        elif act in ['sit', 'step', 'walk']:            return 0
        return -1

    df['label'] = df['label'].apply(label_map)
    df = df[df['label'] != -1]
    data = df[FEATURES + ['accel_mag', 'label']].values

    for i in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
        window = data[i: i + WINDOW_SIZE]
        label  = int(np.round(np.mean(window[:, -1])))
        if label == 1 and window[:, -2].max() < 1.5:
            continue
        X_list.append(window[:, :N_FEATURES])
        y_list.append(label)

    return np.array(X_list), np.array(y_list)

print("=" * 50)
print("  FALL DETECTION — DECISION TREE")
print("=" * 50)
print("\n  Loading data ...")
X, y = prepare_data()

if len(X) == 0:
    exit()

print(f"  Windows: {X.shape[0]}  |  Fall: {y.sum()}  |  Normal: {(y==0).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Flatten for sklearn
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train.reshape(-1, N_FEATURES)).reshape(X_train.shape[0], -1)
X_test_s  = scaler.transform(X_test.reshape(-1, N_FEATURES)).reshape(X_test.shape[0], -1)

print(f"\n  Input shape (flattened): {X_train_s.shape}")

# ─────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────
model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)

print("\n  Training ...")
model.fit(X_train_s, y_train)

print(f"\n  Tree depth : {model.get_depth()}")
print(f"  Leaf nodes : {model.get_n_leaves()}")

y_pred = model.predict(X_test_s)

print("\n" + "=" * 50)
print("  RESULTS — DECISION TREE")
print("=" * 50)
print(f"  Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"  F1-Score : {f1_score(y_test, y_pred)*100:.2f}%")
print("\n" + classification_report(y_test, y_pred, target_names=['Normal', 'Fall']))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Decision Tree — Fall Detection', fontsize=14, fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='BuGn', ax=axes[0],
            xticklabels=['Normal', 'Fall'], yticklabels=['Normal', 'Fall'])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')

# Top-20 Feature Importances
importances  = model.feature_importances_
feature_names = []
for t in range(WINDOW_SIZE):
    for f in FEATURES:
        feature_names.append(f"{f}_t{t}")

top_idx   = np.argsort(importances)[::-1][:20]
top_vals  = importances[top_idx]
top_names = [feature_names[i] for i in top_idx]

axes[1].barh(range(20), top_vals[::-1], color='seagreen')
axes[1].set_yticks(range(20))
axes[1].set_yticklabels(top_names[::-1], fontsize=7)
axes[1].set_title('Top 20 Feature Importances')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('results_decision_tree.png', dpi=150, bbox_inches='tight')
plt.show()

# Visualise top levels of the tree (depth ≤ 4 for readability)
plt.figure(figsize=(20, 8))
plot_tree(model, max_depth=4, filled=True,
          class_names=['Normal', 'Fall'],
          feature_names=feature_names,
          fontsize=6, proportion=True)
plt.title('Decision Tree Structure (Top 4 Levels)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results_decision_tree_structure.png', dpi=120, bbox_inches='tight')
plt.show()

joblib.dump(model,  'model_decision_tree.pkl')
joblib.dump(scaler, 'scaler_decision_tree.pkl')
print("\n Saved: model_decision_tree.pkl | scaler_decision_tree.pkl")
print("         results_decision_tree.png | results_decision_tree_structure.png")
