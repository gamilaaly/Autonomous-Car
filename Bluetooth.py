import serial
import time
s = serial.Serial('COM10', 9600,timeout = 1) # choose the outgoing one
print("connected!")
time.sleep(2)
s.write(b"f")
print("Sent Message!")