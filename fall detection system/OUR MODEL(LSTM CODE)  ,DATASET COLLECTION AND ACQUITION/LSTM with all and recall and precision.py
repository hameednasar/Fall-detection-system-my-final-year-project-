import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score  # ← added PR imports

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L2

FILE_PATH = r"C:\Users\p.c\Desktop\Dataset acquition\Ourdataset.csv"

WINDOW_SIZE = 40
STEP_SIZE = 5

FEATURES = ['xAcc', 'yAcc', 'zAcc', 'xGyro', 'yGyro', 'zGyro']

def prepare_data():
    X_list, y_list = [], []
    
    if not os.path.exists(FILE_PATH):
        print(" File not found!")
        return np.array([]), np.array([])
    
    df = pd.read_csv(FILE_PATH)

    accel_mag = np.sqrt(df['xAcc']**2 + df['yAcc']**2 + df['zAcc']**2)
    df['accel_mag'] = accel_mag

    def label_map(act):
        if act in ['fall', 'lfall', 'rfall', 'light','ffall']:
            return 1
        elif act in ['sit', 'step', 'walk']:
            return 0
        else:
            return -1

    df['label'] = df['label'].apply(label_map)
    df = df[df['label'] != -1]

    data = df[FEATURES + ['accel_mag', 'label']].values

    for i in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
        window = data[i : i + WINDOW_SIZE]

        labels = window[:, -1]
        label = int(np.round(np.mean(labels)))

        if label == 1:
            if window[:, -2].max() < 1.5:
                continue

        X_list.append(window[:, :-2])
        y_list.append(label)

    return np.array(X_list), np.array(y_list)

print(" Processing data and filtering noise...")
X, y = prepare_data()

if len(X) == 0:
    print(" No data found! Check dataset.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 6)).reshape(-1, WINDOW_SIZE, 6)
X_test_scaled = scaler.transform(X_test.reshape(-1, 6)).reshape(-1, WINDOW_SIZE, 6)

noise_level = 0.02
X_train_noisy = X_train_scaled + np.random.normal(0, noise_level, X_train_scaled.shape)

model = Sequential([
    Input(shape=(WINDOW_SIZE, 6)),
    LSTM(64, return_sequences=True, kernel_regularizer=L2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    
    LSTM(32, kernel_regularizer=L2(0.001)),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(16, activation='relu', kernel_regularizer=L2(0.001)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print(" Training model...")
history = model.fit(
    X_train_noisy, y_train, 
    epochs=40, 
    batch_size=32, 
    validation_data=(X_test_scaled, y_test),
    callbacks=[early_stop]
)

print("\n" + "="*50)
print(" FINAL EVALUATION")
print("="*50)

loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

y_pred_probs = model.predict(X_test_scaled)
y_pred = (y_pred_probs > 0.5).astype("int32")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal Activity', 'Fall']))

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy Progress')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss Progress')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Normal', 'Predicted Fall'],
            yticklabels=['Actual Normal', 'Actual Fall'])

plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Classifier')
plt.fill_between(fpr, tpr, alpha=0.1, color='darkorange')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC — AUC Curve (LSTM Fall Detection)', fontsize=14)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_auc_lstm.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n AUC Score: {roc_auc:.4f}")

precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_probs)
avg_precision = average_precision_score(y_test, y_pred_probs)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', lw=2,
         label=f'PR Curve (AP = {avg_precision:.4f})')
plt.fill_between(recall, precision, alpha=0.1, color='green')
plt.axhline(y=sum(y_test) / len(y_test), color='navy', lw=1.5,
            linestyle='--', label='Random Classifier (Baseline)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve (LSTM Fall Detection)', fontsize=14)
plt.legend(loc='lower left', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('precision_recall_lstm.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n Average Precision (AP) Score: {avg_precision:.4f}")

model.save('fall_detection_model.keras')
joblib.dump(scaler, 'fall_detection_scaler.pkl')

print(" Done! Model and Scaler saved.")