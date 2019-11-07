import serial
import time
s = serial.Serial('COM10', 9600,timeout = 1) # choose the outgoing one
print("connected!")
time.sleep(2)
s.write(b"f")
print("forward")
time.sleep(5)
s.write(b"b")
print("backward!")
time.sleep(5)
s.write(b"l")
print("left")
time.sleep(5)
s.write(b"r")
print("right")
time.sleep(5)
s.write(b"s")
print("stop")
time.sleep(5)





