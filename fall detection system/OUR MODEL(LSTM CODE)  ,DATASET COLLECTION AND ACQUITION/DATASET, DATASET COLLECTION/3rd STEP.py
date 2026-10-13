import serial
import csv
import re
import time
PORT = 'COM4' 
BAUD = 115200

ACCEL_SENSITIVITY = 16384.0  # LSB/g
GYRO_SENSITIVITY = 131.0     # LSB/deg/s
GRAVITY = 9.81               # m/s^2

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2) 
    print(f"--- Connected to {PORT} | Recording SCALED Data ---")

    # Create/Open CSV
    with open('ffall.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # Headers match the "Points" format you want
        writer.writerow(['xAcc', 'yAcc', 'zAcc', 'xGyro', 'yGyro', 'zGyro'])

        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            if line:
                # Find all integers (including negatives)
                numbers = re.findall(r'-?\d+', line)

                if len(numbers) == 6:
                    raw = [int(n) for n in numbers]
                    
                    # 1. Scale Accelerometer to m/s^2
                    xAcc = round((raw[0] / ACCEL_SENSITIVITY) * GRAVITY, 2)
                    yAcc = round((raw[1] / ACCEL_SENSITIVITY) * GRAVITY, 2)
                    zAcc = round((raw[2] / ACCEL_SENSITIVITY) * GRAVITY, 2)
                    
                    # 2. Scale Gyroscope to deg/s
                    xGyro = round(raw[3] / GYRO_SENSITIVITY, 2)
                    yGyro = round(raw[4] / GYRO_SENSITIVITY, 2)
                    zGyro = round(raw[5] / GYRO_SENSITIVITY, 2)

                    scaled_row = [xAcc, yAcc, zAcc, xGyro, yGyro, zGyro]
                    
                    # Print to console to verify the "points" look correct
                    print(f"Points: {scaled_row}")
                    
                    writer.writerow(scaled_row)
                    file.flush() 

except serial.SerialException as e:
    print(f"Error: {e}. Check if Serial Monitor is open!")
except KeyboardInterrupt:
    print("\nData collection stopped. CSV saved.")
finally:
    if 'ser' in locals():
        ser.close()