import requests
import face_recognition
import picamera
import time
import numpy as np

URL = 'http://192.168.0.13:8080/image'

def capture_and_send():
    camera = picamera.PiCamera()
    rgbimage = np.empty((320, 480, 3), dtype = np.uint8)
    camera.resolution = (480,320)
    camera.start_preview()
    time.sleep(5)
    while(1):
        time.sleep(1)
        camera.capture(rgbimage,format="rgb")
        face_locations = face_recognition.face_locations(rgbimage)
        face_encodings = face_recognition.face_encodings(rgbimage,face_locations)
        print (time.time())

if __name__ == "__main__":
    capture_and_send()
