

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (LSTM, Conv1D, MaxPooling1D, Flatten,
                                     Dense, Dropout, BatchNormalization,
                                     Input, TimeDistributed)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L2

FILE_PATH   = r"C:\Users\p.c\Desktop\Data collection\acc_gyr.csv"
WINDOW_SIZE = 40
STEP_SIZE   = 5
FEATURES    = ['xAcc', 'yAcc', 'zAcc', 'xGyro', 'yGyro', 'zGyro']
N_FEATURES  = 6
NOISE_LEVEL = 0.02

N_STEPS = 4                        
N_LEN   = WINDOW_SIZE // N_STEPS   

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
print("  FALL DETECTION — CNN-LSTM")
print("=" * 50)
print("\n  Loading data ...")
X, y = prepare_data()

if len(X) == 0:
    exit()

print(f"  Windows: {X.shape[0]}  |  Fall: {y.sum()}  |  Normal: {(y==0).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train.reshape(-1, N_FEATURES)).reshape(-1, WINDOW_SIZE, N_FEATURES)
X_test_s  = scaler.transform(X_test.reshape(-1, N_FEATURES)).reshape(-1, WINDOW_SIZE, N_FEATURES)
X_train_noisy = X_train_s + np.random.normal(0, NOISE_LEVEL, X_train_s.shape)

# Reshape into (samples, N_STEPS, N_LEN, N_FEATURES) for CNN-LSTM
X_train_r = X_train_noisy.reshape(-1, N_STEPS, N_LEN, N_FEATURES)
X_test_r  = X_test_s.reshape(-1, N_STEPS, N_LEN, N_FEATURES)

print(f"\n  CNN-LSTM input shape: {X_train_r.shape}")
print(f"  Sub-sequences: {N_STEPS}  |  Sub-seq length: {N_LEN}")

model = Sequential([
    Input(shape=(N_STEPS, N_LEN, N_FEATURES)),

    TimeDistributed(Conv1D(64, kernel_size=3, activation='relu', padding='same')),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(MaxPooling1D(pool_size=2)),
    TimeDistributed(Dropout(0.3)),

    TimeDistributed(Conv1D(128, kernel_size=3, activation='relu', padding='same')),
    TimeDistributed(BatchNormalization()),
    TimeDistributed(Dropout(0.3)),

    TimeDistributed(Flatten()),   

    LSTM(64, return_sequences=False, kernel_regularizer=L2(0.001)),
    BatchNormalization(),
    Dropout(0.5),

    Dense(32, activation='relu', kernel_regularizer=L2(0.001)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
], name='CNN_LSTM')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("\n  Training ...")
history = model.fit(
    X_train_r, y_train,
    epochs=40, batch_size=32,
    validation_data=(X_test_r, y_test),
    callbacks=[early_stop]
)


y_pred = (model.predict(X_test_r) > 0.5).astype(int).flatten()

print("\n" + "=" * 50)
print("  RESULTS — CNN-LSTM")
print("=" * 50)
print(f"  Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"  F1-Score : {f1_score(y_test, y_pred)*100:.2f}%")
print("\n" + classification_report(y_test, y_pred, target_names=['Normal', 'Fall']))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('CNN-LSTM — Fall Detection', fontsize=14, fontweight='bold')

axes[0].plot(history.history['accuracy'],     label='Train')
axes[0].plot(history.history['val_accuracy'], label='Val')
axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch')
axes[0].legend()

axes[1].plot(history.history['loss'],     label='Train')
axes[1].plot(history.history['val_loss'], label='Val')
axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch')
axes[1].legend()

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[2],
            xticklabels=['Normal', 'Fall'], yticklabels=['Normal', 'Fall'])
axes[2].set_title('Confusion Matrix')
axes[2].set_xlabel('Predicted'); axes[2].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('results_cnn_lstm.png', dpi=150, bbox_inches='tight')
plt.show()

model.save('model_cnn_lstm.keras')
joblib.dump(scaler, 'scaler_cnn_lstm.pkl')
print("\n  Saved: model_cnn_lstm.keras | scaler_cnn_lstm.pkl | results_cnn_lstm.png")
