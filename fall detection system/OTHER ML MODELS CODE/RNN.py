

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
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L2

FILE_PATH   = r"C:\Users\p.c\Desktop\Other model training\acc_gyr.csv"
WINDOW_SIZE = 40
STEP_SIZE   = 5
FEATURES    = ['xAcc', 'yAcc', 'zAcc', 'xGyro', 'yGyro', 'zGyro']
N_FEATURES  = 6
NOISE_LEVEL = 0.02

def prepare_data():
    X_list, y_list = [], []
    if not os.path.exists(FILE_PATH):
        print("  File not found:", FILE_PATH)
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

print("  FALL DETECTION — RNN")
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

print(f"\n  Training samples : {X_train_s.shape[0]}")
print(f"  Test samples     : {X_test_s.shape[0]}")
print(f"  Input shape      : {X_train_s.shape[1:]}")


model = Sequential([
    Input(shape=(WINDOW_SIZE, N_FEATURES)),

    # First SimpleRNN layer — returns sequences for stacking
    SimpleRNN(64, return_sequences=True,
              activation='tanh',
              kernel_regularizer=L2(0.001),
              recurrent_regularizer=L2(0.001)),
    BatchNormalization(),
    Dropout(0.5),

    SimpleRNN(32, return_sequences=True,
              activation='tanh',
              kernel_regularizer=L2(0.001),
              recurrent_regularizer=L2(0.001)),
    BatchNormalization(),
    Dropout(0.5),

    SimpleRNN(16, return_sequences=False,
              activation='tanh',
              kernel_regularizer=L2(0.001)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(32, activation='relu', kernel_regularizer=L2(0.001)),
    Dropout(0.3),
    Dense(16, activation='relu', kernel_regularizer=L2(0.001)),
    Dense(1,  activation='sigmoid')
], name='RNN')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

print("\n  Training ...")
history = model.fit(
    X_train_noisy, y_train,
    epochs=40,
    batch_size=32,
    validation_data=(X_test_s, y_test),
    callbacks=[early_stop]
)

y_pred_prob = model.predict(X_test_s, verbose=0)
y_pred      = (y_pred_prob > 0.5).astype(int).flatten()

print("\n" + "=" * 50)
print("  RESULTS — RNN")
print("=" * 50)
print(f"  Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"  F1-Score  : {f1_score(y_test, y_pred)*100:.2f}%")
print("\n" + classification_report(y_test, y_pred, target_names=['Normal', 'Fall']))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('RNN — Fall Detection', fontsize=14, fontweight='bold')

# Accuracy curve
axes[0].plot(history.history['accuracy'],     label='Train', color='steelblue', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val',   color='coral',     linewidth=2)
axes[0].set_title('Model Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train', color='steelblue', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Val',   color='coral',     linewidth=2)
axes[1].set_title('Model Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=axes[2],
            xticklabels=['Normal', 'Fall'],
            yticklabels=['Normal', 'Fall'],
            annot_kws={"size": 14})
axes[2].set_title('Confusion Matrix')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('results_rnn.png', dpi=150, bbox_inches='tight')
plt.show()

model.save('model_rnn.keras')
joblib.dump(scaler, 'scaler_rnn.pkl')
print("\n  Saved: model_rnn.keras | scaler_rnn.pkl | results_rnn.png")