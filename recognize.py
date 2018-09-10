import face_recognition
import numpy as np
import os
import time

known_face_encodings = []
known_face_names = []

def read_files_in_dir(dir_path):
    for root,dirs,files in os.walk(dir_path):
        for fname in files:
            image = np.load(dir_path + fname)
            known_face_encodings.append(image)
            known_face_names.append(fname.replace('.npy',''))
def matching_file(npy_file):
    matches = face_recognition.compare_faces(known_face_encodings,npy_file,tolerance = 0.5)
    name = "Unkown"

    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
        distance = np.linalg.norm(known_face_encodings[first_match_index]-npy_file)
        print (name,distance)


if __name__ == "__main__":
    read_files_in_dir('./encodings/')
    print ("ready")
    file_dir = input()
    npy_image = np.load(file_dir)
    matching_file(npy_image)
