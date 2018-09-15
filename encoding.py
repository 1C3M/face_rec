#!/usr/bin/python3
import face_recognition
from PIL import Image
import numpy as np
import os

def load_image( infilename ):
    img = Image.open( infilename )
    return np.asarray(img)

def get_encoding( image ):
    face_locations = face_recognition.face_locations(rgbimage)
    face_encodings = face_recognition.face_encodings(rgbimage, face_locations)
    return face_encodings[0]

if __name__=="__main__":
    npyname = input()
    filename = input()

    rgbimage = load_image(filename)
    encoding = get_encoding(rgbimage)
    np.save(npyname, encoding)

    print("success", npyname)
