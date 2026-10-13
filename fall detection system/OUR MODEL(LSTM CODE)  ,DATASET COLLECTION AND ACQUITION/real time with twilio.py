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
from twilio.rest import Client

PORT            = 'COM4'
BAUD            = 115200
MODEL_PATH      = r'C:\Users\p.c\Desktop\Dataset acquition\fall_detection_model.keras'
SCALER_PATH     = r'C:\Users\p.c\Desktop\Dataset acquition\fall_detection_scaler.pkl'
CSV_PATH        = r'C:\Users\p.c\Desktop\Dataset acquition\imu_data_scaled.csv'
WINDOW_SIZE     = 150
PREDICT_EVERY   = 10

TWILIO_ACCOUNT_SID = ''   
TWILIO_AUTH_TOKEN  = ''                
TWILIO_FROM_NUMBER = '+12298608364'                         
ALERT_TO_NUMBER    = '+923101818092'                        

SMS_DELAY_SECONDS  = 7    
CALL_DELAY_SECONDS = 30   

ACCEL_SENSITIVITY = 16384.0
GYRO_SENSITIVITY  = 131.0
GRAVITY           = 9.81

FEATURES = ['xAcc', 'yAcc', 'zAcc', 'xGyro', 'yGyro', 'zGyro']


print("Loading model and scaler...")
model  = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print(f"Model input shape: {model.input_shape}")
print("Model and scaler loaded successfully!")

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

VOICE_MESSAGE = """
<Response>
    <Say voice="alice" language="en-US">
        Emergency Alert! Emergency Alert!
        A fall has been detected and the patient has been on the ground for over 30 seconds.
        Please check on the patient immediately.
        This is an automated call from the Fall Detection System.
    </Say>
    <Pause length="1"/>
    <Say voice="alice" language="en-US">
        Repeating: The patient has fallen and has not gotten up for over 30 seconds.
        Please respond immediately.
    </Say>
</Response>
"""

def send_sms_alert():
    """Send SMS to caretaker."""
    try:
        message = twilio_client.messages.create(
            body=(
                "FALL ALERT: A fall has been detected!\n"
                "The patient has been on the ground for over 7 seconds.\n"
                "Please check on them immediately."
            ),
            from_=TWILIO_FROM_NUMBER,
            to=ALERT_TO_NUMBER
        )
        print(f"[SMS] Sent successfully! SID: {message.sid}")
    except Exception as e:
        print(f"[SMS] Failed to send: {e}")

def make_call_alert():
    """Make automated voice call to caretaker."""
    try:
        call = twilio_client.calls.create(
            twiml=VOICE_MESSAGE,
            from_=TWILIO_FROM_NUMBER,
            to=ALERT_TO_NUMBER
        )
        print(f"[CALL] Call initiated! SID: {call.sid}")
    except Exception as e:
        print(f"[CALL] Failed to call: {e}")

window      = deque(maxlen=WINDOW_SIZE)
row_counter = 0
prediction  = "Collecting data..."
confidence  = 0.0
state_lock  = threading.Lock()

fall_start_time  = None   
sms_sent         = False  
call_sent        = False  
alert_status_msg = ""     

def extract_features(window_data: np.ndarray) -> np.ndarray:
    flat   = window_data.reshape(-1, window_data.shape[-1])
    scaled = scaler.transform(flat)
    expected_input = model.input_shape
    if len(expected_input) == 3:
        return scaled.reshape(1, WINDOW_SIZE, len(FEATURES))
    else:
        return scaled.flatten().reshape(1, -1)

def update_fall_timer(is_fall: bool):
    global fall_start_time, sms_sent, call_sent, alert_status_msg

    now = time.time()

    with state_lock:
        if is_fall:
            if fall_start_time is None:
                fall_start_time = now
                print(f"[TIMER] Fall started — SMS in {SMS_DELAY_SECONDS}s, Call in {CALL_DELAY_SECONDS}s")

            elapsed = now - fall_start_time

            if not sms_sent and elapsed >= SMS_DELAY_SECONDS:
                sms_sent = True
                alert_status_msg = "Sending SMS..."
                print(f"[TIMER] {elapsed:.1f}s elapsed — firing SMS!")
                threading.Thread(target=_do_sms, daemon=True).start()

            if not call_sent and elapsed >= CALL_DELAY_SECONDS:
                call_sent = True
                alert_status_msg = "Calling caretaker..."
                print(f"[TIMER] {elapsed:.1f}s elapsed — firing CALL!")
                threading.Thread(target=_do_call, daemon=True).start()

        else:
            if fall_start_time is not None:
                print("[TIMER] Fall cleared — resetting all alerts.")
            fall_start_time  = None
            sms_sent         = False
            call_sent        = False
            alert_status_msg = ""

def _do_sms():
    global alert_status_msg
    send_sms_alert()
    with state_lock:
        if not call_sent:
            alert_status_msg = "SMS Sent! Watching for 30s call..."

def _do_call():
    global alert_status_msg
    make_call_alert()
    with state_lock:
        alert_status_msg = "Call Made! Both alerts sent."

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

                        if raw_pred.shape[-1] == 1:
                            prob_fall  = float(raw_pred[0][0])
                            pred_class = 1 if prob_fall >= 0.5 else 0
                            conf = round(prob_fall * 100 if pred_class == 1
                                         else (1 - prob_fall) * 100, 1)
                        else:
                            pred_class = int(np.argmax(raw_pred[0]))
                            conf = round(float(np.max(raw_pred[0])) * 100, 1)

                        label = "FALL DETECTED" if pred_class == 1 else "No Fall"

                        with state_lock:
                            prediction = label
                            confidence = conf

                        print(f"[Row {current_count:>6}]  {label}  ({conf}% confidence)")

                        update_fall_timer(is_fall=(pred_class == 1))

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
        root.geometry("560x460")
        root.configure(bg="#1a1a2e")
        root.resizable(False, False)

        title_font  = tkfont.Font(family="Helvetica", size=16, weight="bold")
        result_font = tkfont.Font(family="Helvetica", size=26, weight="bold")
        info_font   = tkfont.Font(family="Helvetica", size=11)
        alert_font  = tkfont.Font(family="Helvetica", size=12, weight="bold")
        small_font  = tkfont.Font(family="Helvetica", size=9)

        tk.Label(root, text="MPU6050 Fall Detection System",
                 font=title_font, bg="#1a1a2e", fg="#e0e0e0").pack(pady=(20, 4))

        tk.Label(root,
                 text=f"Window: {WINDOW_SIZE} rows  |  Model: {type(model).__name__}  |  Port: {PORT}",
                 font=info_font, bg="#1a1a2e", fg="#666666").pack()

        self.status_frame = tk.Frame(root, bg="#16213e",
                                     highlightbackground="#0f3460",
                                     highlightthickness=2)
        self.status_frame.pack(fill="x", padx=40, pady=14)

        self.result_label = tk.Label(self.status_frame, text="Collecting data...",
                                     font=result_font, bg="#16213e", fg="#e0e0e0", pady=16)
        self.result_label.pack()

        self.conf_label = tk.Label(self.status_frame, text="",
                                   font=info_font, bg="#16213e", fg="#aaaaaa", pady=2)
        self.conf_label.pack()

        self.timer_label = tk.Label(self.status_frame, text="",
                                    font=info_font, bg="#16213e", fg="#ffaa00", pady=4)
        self.timer_label.pack()

        # ── Alert stages indicator ──
        stages_frame = tk.Frame(root, bg="#1a1a2e")
        stages_frame.pack(fill="x", padx=40, pady=(4, 2))

        self.sms_indicator = tk.Label(stages_frame,
                                      text=f"SMS  @ {SMS_DELAY_SECONDS}s",
                                      font=info_font, bg="#222244", fg="#777777",
                                      width=18, pady=5)
        self.sms_indicator.pack(side="left", expand=True, fill="x", padx=(0, 4))

        self.call_indicator = tk.Label(stages_frame,
                                       text=f"Call @ {CALL_DELAY_SECONDS}s",
                                       font=info_font, bg="#222244", fg="#777777",
                                       width=18, pady=5)
        self.call_indicator.pack(side="left", expand=True, fill="x", padx=(4, 0))

        self.alert_label = tk.Label(root, text="",
                                    font=alert_font, bg="#1a1a2e", fg="#00ffaa", pady=4)
        self.alert_label.pack()

        self.rows_label = tk.Label(root, text="Rows collected: 0  |  Buffer: 0/150",
                                   font=info_font, bg="#1a1a2e", fg="#777777")
        self.rows_label.pack(pady=(2, 0))

        tk.Label(root, text=f"Caretaker: {ALERT_TO_NUMBER}",
                 font=small_font, bg="#1a1a2e", fg="#555555").pack(pady=(6, 0))

        tk.Label(root, text="Ctrl+C in terminal to stop  |  Close window to exit",
                 font=small_font, bg="#1a1a2e", fg="#444444").pack(pady=(3, 0))

        self.update_gui()

    def update_gui(self):
        with state_lock:
            pred       = prediction
            conf       = confidence
            rows       = row_counter
            buf        = len(window)
            f_start    = fall_start_time
            s_sent     = sms_sent
            c_sent     = call_sent
            status_msg = alert_status_msg

        # Main status colours
        if "FALL" in pred:
            color, bg = "#ff4d4d", "#2d0000"
        elif "No Fall" in pred:
            color, bg = "#4dff88", "#002d11"
        else:
            color, bg = "#e0e0e0", "#16213e"

        self.status_frame.configure(bg=bg)
        self.result_label.configure(text=pred, fg=color, bg=bg)
        self.conf_label.configure(bg=bg, text=f"Confidence: {conf}%" if conf else "")

        # Countdown timer
        if f_start is not None and "FALL" in pred:
            elapsed = time.time() - f_start
            if not c_sent:
                remaining_call = max(0, CALL_DELAY_SECONDS - elapsed)
                if not s_sent:
                    remaining_sms = max(0, SMS_DELAY_SECONDS - elapsed)
                    timer_text = f"SMS in {remaining_sms:.1f}s  |  Call in {remaining_call:.1f}s"
                else:
                    timer_text = f"SMS sent  |  Call in {remaining_call:.1f}s"
            else:
                timer_text = f"All alerts sent  |  Down: {elapsed:.0f}s"
            self.timer_label.configure(text=timer_text, bg=bg, fg="#ffaa00")
        else:
            self.timer_label.configure(text="", bg=bg)

        # Stage indicators
        sms_color  = "#00cc66" if s_sent else ("#ffaa00" if (f_start and not s_sent) else "#777777")
        sms_bg     = "#003311" if s_sent else "#222244"
        call_color = "#ff6600" if c_sent else ("#ffaa00" if (f_start and s_sent and not c_sent) else "#777777")
        call_bg    = "#331100" if c_sent else "#222244"

        self.sms_indicator.configure(fg=sms_color, bg=sms_bg,
                                     text=f"SMS  @ {SMS_DELAY_SECONDS}s {'(Sent)' if s_sent else ''}")
        self.call_indicator.configure(fg=call_color, bg=call_bg,
                                      text=f"Call @ {CALL_DELAY_SECONDS}s {'(Done)' if c_sent else ''}")

        self.alert_label.configure(text=status_msg)
        self.rows_label.configure(text=f"Rows collected: {rows}  |  Buffer: {buf}/{WINDOW_SIZE}")

        self.root.after(300, self.update_gui)

if __name__ == "__main__":
    t = threading.Thread(target=serial_thread, daemon=True)
    t.start()

    root = tk.Tk()
    app  = FallDetectionGUI(root)
    root.mainloop()
