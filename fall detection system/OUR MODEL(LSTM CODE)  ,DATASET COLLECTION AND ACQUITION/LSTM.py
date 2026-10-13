import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L2

FILE_PATH =r"C:\Users\p.c\Desktop\Dataset acquition\kaggle_dataset.csv"

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

    # LABEL MAPPING 
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

        # Noise filtering 
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

# Add noise to training data for better generalization
noise_level = 0.02
X_train_noisy = X_train_scaled + np.random.normal(0, noise_level, X_train_scaled.shape)

# MODEL ARCHITECTURE
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

# EVALUATION
print("\n" + "="*50)
print(" FINAL EVALUATION")
print("="*50)

loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

y_pred_probs = model.predict(X_test_scaled)
y_pred = (y_pred_probs > 0.5).astype("int32")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal Activity', 'Fall']))

# VISUALIZATION
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

model.save('fall_detection_model.keras')
joblib.dump(scaler, 'fall_detection_scaler.pkl')

print(" Done! Model and Scaler saved.")