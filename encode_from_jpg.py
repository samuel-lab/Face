import face_recognition
import hashlib

try:
    new_face = face_recognition.load_image_file('newface.jpg')
    new_face_encoding = face_recognition.face_encodings(new_face)[0]
    #print(new_face_encoding)
    new_face_encoding_hash = hashlib.sha256(new_face_encoding).hexdigest()
    print(new_face_encoding_hash)

except:
    print("error")

try: 
    new_face2 = face_recognition.load_image_file('Unknown_Person.jpg')
    new_face_encoding2 = face_recognition.face_encodings(new_face2)[0]
    #print(new_face_encoding)
    new_face_encoding_hash2 = hashlib.sha256(new_face_encoding2).hexdigest()
    print(new_face_encoding_hash2)

except:
    print("error")
if new_face_encoding_hash == new_face_encoding_hash2:
    print("Match")
else:
    print("Not Match")