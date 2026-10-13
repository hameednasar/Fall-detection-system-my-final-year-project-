import serial
import csv
import re
import time
import joblib
import numpy as np
from collections import deque
import threading
import tkinter as tk
from tkinter import font as tkfont
from tensorflow.keras.models import load_model

PORT            = 'COM4'
BAUD            = 115200
MODEL_PATH      = r'C:\Users\p.c\Desktop\Dataset acquition\fall_detection_model.keras'
SCALER_PATH     = r'C:\Users\p.c\Desktop\Dataset acquition\fall_detection_scaler.pkl'
CSV_PATH        = r'C:\Users\p.c\Desktop\Dataset acquition\imu_data_scaled.csv'
WINDOW_SIZE     = 150
PREDICT_EVERY   = 10

ACCEL_SENSITIVITY = 16384.0
GYRO_SENSITIVITY  = 131.0
GRAVITY           = 9.81

FEATURES = ['xAcc', 'yAcc', 'zAcc', 'xGyro', 'yGyro', 'zGyro']

print("Loading model and scaler...")
model  = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print(f"Model input shape: {model.input_shape}")
print("Model and scaler loaded successfully!")

window      = deque(maxlen=WINDOW_SIZE)
row_counter = 0
prediction  = "Collecting data..."
confidence  = 0.0
state_lock  = threading.Lock()

def extract_features(window_data: np.ndarray) -> np.ndarray:
    """
    Scale and reshape window to match model input.
    Handles both flat (ML) and 3D (LSTM/CNN) input shapes automatically.
    """
    flat   = window_data.reshape(-1, window_data.shape[-1])   
    scaled = scaler.transform(flat)        
    expected_input = model.input_shape  
    if len(expected_input) == 3:
        return scaled.reshape(1, WINDOW_SIZE, len(FEATURES))
    else:
        return scaled.flatten().reshape(1, -1)

def serial_thread():
    global row_counter, prediction, confidence

    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2)
        print(f"Connected to {PORT}")
    except serial.SerialException as e:
        print(f"Serial error: {e}. Is COM4 correct and Arduino connected?")
        return

    with open(CSV_PATH, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(FEATURES)

        while True:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if not line:
                    continue

                numbers = re.findall(r'-?\d+', line)
                if len(numbers) != 6:
                    continue

                raw = [int(n) for n in numbers]

                scaled_row = [
                    round((raw[0] / ACCEL_SENSITIVITY) * GRAVITY, 4),
                    round((raw[1] / ACCEL_SENSITIVITY) * GRAVITY, 4),
                    round((raw[2] / ACCEL_SENSITIVITY) * GRAVITY, 4),
                    round(raw[3] / GYRO_SENSITIVITY, 4),
                    round(raw[4] / GYRO_SENSITIVITY, 4),
                    round(raw[5] / GYRO_SENSITIVITY, 4),
                ]

                writer.writerow(scaled_row)
                file.flush()

                with state_lock:
                    window.append(scaled_row)
                    row_counter += 1
                    current_count  = row_counter
                    current_window = list(window)

                if current_count % PREDICT_EVERY == 0 and len(current_window) == WINDOW_SIZE:
                    try:
                        w_array  = np.array(current_window, dtype=np.float32)
                        features = extract_features(w_array)

                        raw_pred = model.predict(features, verbose=0)

                        # Handle sigmoid (binary) vs softmax (multi-class) output
                        if raw_pred.shape[-1] == 1:
                            # Sigmoid output → single probability of fall
                            prob_fall = float(raw_pred[0][0])
                            pred_class = 1 if prob_fall >= 0.5 else 0
                            conf = round(prob_fall * 100 if pred_class == 1
                                         else (1 - prob_fall) * 100, 1)
                        else:
                            # Softmax output → [prob_no_fall, prob_fall]
                            pred_class = int(np.argmax(raw_pred[0]))
                            conf = round(float(np.max(raw_pred[0])) * 100, 1)

                        label = " FALL DETECTED" if pred_class == 1 else " No Fall"

                        with state_lock:
                            prediction = label
                            confidence = conf

                        print(f"[Row {current_count:>6}]  {label}  ({conf}% confidence)")

                    except Exception as pred_err:
                        print(f"Prediction error: {pred_err}")

            except serial.SerialException:
                print("Serial connection lost.")
                break
            except KeyboardInterrupt:
                break

    ser.close()
    print("Serial port closed.")

class FallDetectionGUI:
    def __init__(self, root):
        self.root = root
        root.title("Fall Detection — Real-Time")
        root.geometry("540x360")
        root.configure(bg="#1a1a2e")
        root.resizable(False, False)

        title_font  = tkfont.Font(family="Helvetica", size=16, weight="bold")
        result_font = tkfont.Font(family="Helvetica", size=26, weight="bold")
        info_font   = tkfont.Font(family="Helvetica", size=11)
        small_font  = tkfont.Font(family="Helvetica", size=9)

        tk.Label(root, text="MPU6050 Fall Detection System",
                 font=title_font, bg="#1a1a2e", fg="#e0e0e0").pack(pady=(22, 4))

        tk.Label(root,
                 text=f"Window: {WINDOW_SIZE} rows  |  Model: {type(model).__name__}  |  Port: {PORT}",
                 font=info_font, bg="#1a1a2e", fg="#666666").pack()

        # Status box
        self.status_frame = tk.Frame(root, bg="#16213e",
                                     highlightbackground="#0f3460",
                                     highlightthickness=2)
        self.status_frame.pack(fill="x", padx=40, pady=18)

        self.result_label = tk.Label(self.status_frame,
                                     text="Collecting data...",
                                     font=result_font,
                                     bg="#16213e", fg="#e0e0e0",
                                     pady=18)
        self.result_label.pack()

        self.conf_label = tk.Label(self.status_frame, text="",
                                   font=info_font, bg="#16213e",
                                   fg="#aaaaaa", pady=4)
        self.conf_label.pack()

        self.rows_label = tk.Label(root, text="Rows collected: 0  |  Buffer: 0/150",
                                   font=info_font, bg="#1a1a2e", fg="#777777")
        self.rows_label.pack(pady=(4, 0))

        tk.Label(root, text="Ctrl+C in terminal to stop  •  Close window to exit",
                 font=small_font, bg="#1a1a2e", fg="#444444").pack(pady=(8, 0))

        self.update_gui()

    def update_gui(self):
        with state_lock:
            pred  = prediction
            conf  = confidence
            rows  = row_counter
            buf   = len(window)

        if "FALL" in pred:
            color, bg = "#ff4d4d", "#2d0000"
        elif "No Fall" in pred:
            color, bg = "#4dff88", "#002d11"
        else:
            color, bg = "#e0e0e0", "#16213e"

        self.status_frame.configure(bg=bg)
        self.result_label.configure(text=pred, fg=color, bg=bg)
        self.conf_label.configure(bg=bg,
                                  text=f"Confidence: {conf}%" if conf else "")
        self.rows_label.configure(
            text=f"Rows collected: {rows}  |  Buffer: {buf}/{WINDOW_SIZE}")

        self.root.after(300, self.update_gui)

if __name__ == "__main__":
    t = threading.Thread(target=serial_thread, daemon=True)
    t.start()

    root = tk.Tk()
    app  = FallDetectionGUI(root)
    root.mainloop()