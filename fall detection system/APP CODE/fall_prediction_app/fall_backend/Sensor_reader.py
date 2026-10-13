
import serial
import serial.tools.list_ports
import re
import requests
import time
import numpy as np
from collections import deque

BACKEND_URL   = "http://192.168.137.149:8000"   
EMAIL         = "keshav@gmail.com"              
PASSWORD      = "kk123456"                    
BAUD_RATE     = 115200
WINDOW_SIZE   = 150   
PREDICT_EVERY = 10     

ACCEL_SENSITIVITY = 16384.0
GYRO_SENSITIVITY  = 131.0
GRAVITY           = 9.81
SERIAL_PORT = None   

def find_port() -> str | None:
    ports = serial.tools.list_ports.comports()
    print("\n📡 Available serial ports:")
    if not ports:
        print("   (none found)")
    for p in ports:
        print(f"   {p.device}  —  {p.description}")

    for p in ports:
        desc = p.description.lower()
        dev  = p.device
        if any(k in dev for k in ["usbserial", "usbmodem", "ttyUSB", "ttyACM"]):
            print(f"\n Auto-selected: {dev}")
            return dev
        if dev.startswith("COM") and dev != "COM1":   # COM1 is usually system port
            print(f"\n Auto-selected: {dev}")
            return dev

    print("\n  Could not auto-detect ESP32 port.")
    print("   Plug in your sensor, then set SERIAL_PORT manually above.")
    return None


def login() -> str:
    print(f"\n Connecting to backend at {BACKEND_URL} ...")
    while True:
        try:
            # Quick health check first
            hres = requests.get(f"{BACKEND_URL}/health", timeout=3)
            if hres.status_code != 200:
                raise ConnectionError("Backend unhealthy")
            print(" Backend reachable.")

            res = requests.post(
                f"{BACKEND_URL}/auth/login",
                json={"email": EMAIL, "password": PASSWORD},
                timeout=5,
            )
            if res.status_code == 200:
                token = res.json()["access_token"]
                name  = res.json().get("name", EMAIL)
                print(f" Logged in as {name}\n")
                return token
            err = res.json().get("detail", res.text)
            print(f" Login failed: {err}")
            print("   Check EMAIL / PASSWORD in sensor_reader.py")
            print("   Retrying in 5 s...")
        except Exception as e:
            print(f"  Cannot reach backend: {e}")
            print(f"   Make sure backend.py is running at {BACKEND_URL}")
            print("   Retrying in 5 s...")
        time.sleep(5)


def parse_line(line: str) -> list[float] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    numbers = re.findall(r"-?\d+", line)
    if len(numbers) < 6:
        return None
    raw = [int(n) for n in numbers[:6]]
    return [
        round((raw[0] / ACCEL_SENSITIVITY) * GRAVITY, 4),   
        round((raw[1] / ACCEL_SENSITIVITY) * GRAVITY, 4),   
        round((raw[2] / ACCEL_SENSITIVITY) * GRAVITY, 4),   
        round(raw[3] / GYRO_SENSITIVITY, 4),                 
        round(raw[4] / GYRO_SENSITIVITY, 4),                 
        round(raw[5] / GYRO_SENSITIVITY, 4),                 
    ]


def send_prediction(window: list, latest_row: list,
                    row_count: int, token: str) -> dict | None:
    try:
        res = requests.post(
            f"{BACKEND_URL}/predict/fall",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "window":         window,
                "latest_row":     latest_row,   # ← live values for Flutter display
                "rows_collected": row_count,
            },
            timeout=5,
        )
        if res.status_code == 200:
            return res.json()
        if res.status_code == 503:
            print("\n⚠️  Model not loaded on backend.")
            print("   Place fall_detection_model.keras + fall_detection_scaler.pkl")
            print("   next to backend.py and restart backend.")
            print("   (Sensor data IS being sent and Flutter will show it connected.)")
            return None
        print(f"⚠️  Backend {res.status_code}: {res.text[:120]}")
        return None
    except Exception as e:
        print(f"⚠️  Request failed: {e}")
        return None


def print_result(result: dict, row: int):
    fall  = result.get("fall_detected", False)
    conf  = result.get("confidence", 0) * 100
    risk  = result.get("risk_label", "?")
    icon  = "🚨 FALL DETECTED!" if fall else "✅ Normal activity"
    filled = int(conf / 5)
    bar   = "█" * filled + "░" * (20 - filled)
    print(f"\n[Row {row:>6}]  {icon}")
    print(f"           Risk: {risk}  |  [{bar}] {conf:.1f}%")


def open_serial(port: str, baud: int) -> serial.Serial:
    while True:
        try:
            ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)
            print(f"✅ Serial open: {port} @ {baud} baud")
            return ser
        except serial.SerialException as e:
            print(f"❌ Cannot open {port}: {e}")
            print("   Check cable / port and press Enter to retry...")
            input()


def main():
    port = SERIAL_PORT or find_port()
    if port is None:
        print("\n❌ No serial port found and SERIAL_PORT not set.")
        print("   Connect your ESP32 via USB, set SERIAL_PORT manually,")
        print("   and run again.")
        return

    token       = login()
    window_buf  = deque(maxlen=WINDOW_SIZE)
    row_counter = 0

    print(f"📡 Opening {port} at {BAUD_RATE} baud ...")
    print("   ESP32 must send 6 integers per line, e.g.:")
    print("   1234 -567 16020 45 -12 8")
    print("   Press Ctrl+C to stop.\n")
    print("─" * 65)

    try:
        ser = open_serial(port, BAUD_RATE)

        while True:
            try:
                raw_bytes = ser.readline()
            except serial.SerialException:
                print("\n⚠️  Serial disconnected. Reconnecting...")
                ser.close()
                time.sleep(2)
                ser = open_serial(port, BAUD_RATE)
                continue

            try:
                line = raw_bytes.decode("utf-8", errors="ignore")
            except Exception:
                continue

            row = parse_line(line)

            if row is None:
                stripped = line.strip()
                if stripped:
                    print(f"   [raw – not parsed] {stripped}")
                continue

            window_buf.append(row)
            row_counter += 1

            print(
                f"   xA={row[0]:7.3f}  yA={row[1]:7.3f}  zA={row[2]:7.3f}"
                f"  |  xG={row[3]:7.3f}  yG={row[4]:7.3f}  zG={row[5]:7.3f}"
                f"  buf={len(window_buf)}/{WINDOW_SIZE}",
                end="\r",
            )

            # Predict when buffer is full, every PREDICT_EVERY rows
            if len(window_buf) == WINDOW_SIZE and row_counter % PREDICT_EVERY == 0:
                result = send_prediction(list(window_buf), row, row_counter, token)
                if result:
                    print_result(result, row_counter)

    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user.")
    finally:
        try:
            ser.close()
            print("Serial port closed.")
        except Exception:
            pass


if __name__ == "__main__":
    main()