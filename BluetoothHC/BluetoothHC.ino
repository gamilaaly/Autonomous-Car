int incomingByte = 0;// for incoming serial data
char direction ;

void setup() {
  Serial.begin(9600); // opens serial port, sets data rate to 9600 bps
}

void loop() {
  // send data only when you receive data:
  if (Serial.available() > 0) {
    // read the incoming byte:
  incomingByte = Serial.read();
  if (incomingByte == 108)
  { direction ='1'; // left
    Serial.println(direction);
   }
   else if (incomingByte == 114)
  { direction ='4'; //right
    Serial.println(direction);
   }
   else if (incomingByte == 98)
  {direction ='B';
    Serial.println(direction);
   }
   else if (incomingByte == 102)
  {direction ='F';
    Serial.println(direction);
   }
}}
