from json import JSONEncoder

import face_recognition
import cv2
import numpy as np
import json


# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
def save(face_encode, face_names):
    with open("zaloha/encode.json", "w") as encode:
        okData = {"encode": face_encode}
        #json.dump(okData, encode, cls=NumpyArrayEncoder)
        json.dump(okData, encode, cls=NumpyArrayEncoder, indent = 4)

    with open("zaloha/names.json", "w") as names:
        okData = {"names": face_names}
        json.dump(okData, names, cls=NumpyArrayEncoder, indent = 4)

def load_encode():
    with open("zaloha/encode.json", "r") as outfile:
        encode = json.load(outfile)
        return encode["encode"]

def load_names():
    with open("zaloha/names.json", "r") as outfile:
        names = json.load(outfile)
        return names["names"]

video_capture = cv2.VideoCapture(1)

known_face_encodings = load_encode()
known_face_names = load_names()

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
new_people = 0
last_face = None
max_faces = 1 

while True:
    ret, frame = video_capture.read()

    if process_this_frame:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        if len(face_locations) > max_faces:
            print("Velky pocet osob naraz!")
            cv2.putText(frame, "Veľký počet osôb naraz", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for face_location, face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                else:
                    try:
                        cv2.imwrite('./foto/new_' + str(new_people) + '.jpg', frame)
                        new_face = face_recognition.load_image_file('./foto/new_' + str(new_people) + '.jpg')
                        new_face_encoding = face_recognition.face_encodings(new_face)[0]
                        known_face_encodings.append(new_face_encoding)
                        known_face_names.append('new_' + str(new_people))
                        new_people += 1
                    except:
                        print("new face error")

                face_names.append(name)

    process_this_frame = not process_this_frame

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        save(known_face_encodings, known_face_names)
        break

video_capture.release()
cv2.destroyAllWindows()