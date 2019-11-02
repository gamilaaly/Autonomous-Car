import serial
import time
s = serial.Serial('COM23', 9600,timeout = 1)
print("connected!")
time.sleep(2)
s.write(b"b")
print("Sent Message!")