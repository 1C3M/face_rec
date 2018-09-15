#!/usr/bin/python3
import face_recognition
import numpy as np
import os
import time
import ast

known_face_encodings = []
known_face_names = []

def read_files_in_dir(dir_path):
    for root,dirs,files in os.walk(dir_path):
        for fname in files:
            image = np.load(dir_path + fname)
            known_face_encodings.append(image)
            known_face_names.append(fname.replace('.npy', ''))


def matching_file(encoding):
    matches = face_recognition.compare_faces(known_face_encodings, encoding,tolerance = 0.5)
    name = "Unknown"
    distance = 0

    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]
        distance = np.linalg.norm(known_face_encodings[first_match_index]-encoding)

    return (name,distance)


if __name__ == "__main__":
    npy_path = input()
    read_files_in_dir(npy_path)
    data = np.array(ast.literal_eval(input()))
    matching_result = matching_file(data);
    print(matching_result[0])
    print(matching_result[1])
