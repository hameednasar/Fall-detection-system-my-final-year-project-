"""
=============================================================
  FALL DETECTION — Model 7: SVM (Cubic Kernel)
  Polynomial kernel with degree=3 (cubic) — same as
  MATLAB's "cubic SVM" option.
=============================================================
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
FILE_PATH       = r"C:\Users\p.c\Desktop\Data collection\acc_gyr.csv"
WINDOW_SIZE     = 40
STEP_SIZE       = 5
FEATURES        = ['xAcc', 'yAcc', 'zAcc', 'xGyro', 'yGyro', 'zGyro']
N_FEATURES      = 6
MAX_SVM_SAMPLES = 5000   # SVM is O(n²~³); increase if you have time/RAM

# ─────────────────────────────────────────────
#  DATA PREPARATION
# ─────────────────────────────────────────────
def prepare_data():
    X_list, y_list = [], []
    if not os.path.exists(FILE_PATH):
        print("❌  File not found:", FILE_PATH)
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
print("  FALL DETECTION — SVM (Cubic Kernel)")
print("=" * 50)
print("\n📂  Loading data ...")
X, y = prepare_data()

if len(X) == 0:
    exit()

print(f"✅  Windows: {X.shape[0]}  |  Fall: {y.sum()}  |  Normal: {(y==0).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Flatten for sklearn
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train.reshape(-1, N_FEATURES)).reshape(X_train.shape[0], -1)
X_test_s  = scaler.transform(X_test.reshape(-1, N_FEATURES)).reshape(X_test.shape[0], -1)

print(f"\n  Input shape (flattened): {X_train_s.shape}")

# Sub-sample training set if too large (SVM is slow on large data)
if X_train_s.shape[0] > MAX_SVM_SAMPLES:
    rng = np.random.default_rng(42)
    idx = rng.choice(X_train_s.shape[0], MAX_SVM_SAMPLES, replace=False)
    X_svm = X_train_s[idx]
    y_svm = y_train[idx]
    print(f"  ⚠️  Large dataset detected — training on {MAX_SVM_SAMPLES} samples.")
    print(f"     Increase MAX_SVM_SAMPLES for full training (slower).")
else:
    X_svm = X_train_s
    y_svm = y_train

# ─────────────────────────────────────────────
#  MODEL
#  kernel='poly', degree=3  →  cubic SVM
#  (equivalent to MATLAB's fitcsvm with 'KernelFunction','polynomial','PolynomialOrder',3)
# ─────────────────────────────────────────────
model = SVC(
    kernel='poly',      # polynomial kernel
    degree=3,           # cubic
    C=1.0,              # regularisation (increase for less margin violation)
    gamma='scale',      # kernel coefficient = 1 / (n_features * X.var())
    coef0=1,            # independent term in polynomial kernel
    class_weight='balanced',
    probability=True,   # enable predict_proba (slightly slower)
    random_state=42
)

print("\n🚀  Training SVM — this may take a few minutes ...")
model.fit(X_svm, y_svm)
print("  Training complete.")

# ─────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────
y_pred      = model.predict(X_test_s)
y_pred_prob = model.predict_proba(X_test_s)[:, 1]

print("\n" + "=" * 50)
print("  RESULTS — SVM (Cubic Kernel)")
print("=" * 50)
print(f"  Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"  F1-Score : {f1_score(y_test, y_pred)*100:.2f}%")
print("\n" + classification_report(y_test, y_pred, target_names=['Normal', 'Fall']))

# ─────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('SVM (Cubic Kernel) — Fall Detection', fontsize=14, fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdPu', ax=axes[0],
            xticklabels=['Normal', 'Fall'], yticklabels=['Normal', 'Fall'])
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')

# Prediction probability distribution
axes[1].hist(y_pred_prob[y_test == 0], bins=30, alpha=0.7,
             label='Normal', color='steelblue')
axes[1].hist(y_pred_prob[y_test == 1], bins=30, alpha=0.7,
             label='Fall', color='coral')
axes[1].axvline(0.5, color='black', linestyle='--', label='Threshold = 0.5')
axes[1].set_title('Predicted Probability Distribution')
axes[1].set_xlabel('P(Fall)')
axes[1].set_ylabel('Count')
axes[1].legend()

plt.tight_layout()
plt.savefig('results_svm_cubic.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
#  SAVE
# ─────────────────────────────────────────────
joblib.dump(model,  'model_svm_cubic.pkl')
joblib.dump(scaler, 'scaler_svm_cubic.pkl')
print("\n✅  Saved: model_svm_cubic.pkl | scaler_svm_cubic.pkl | results_svm_cubic.png")
