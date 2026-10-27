from pathlib import Path
import face_recognition
import pickle
from collections import Counter
import cv2 
import numpy as np

DEFAULT_ENCODING_PATH = Path("output/encoding2.pkl")

def _recognize_face(unknown_encoding, known_encodings_data):
    boolean_result = face_recognition.compare_faces(known_encodings_data["encodings"] , unknown_encoding, 0.6)
    # It will give list of true and false for every known_encoding based on it's tolerance match

    votes = Counter(names for result, names in zip(boolean_result, known_encodings_data["names"]) if result)
    # this votes will have a dictionary like counter object => {"names":count}

    if votes:
        return votes.most_common(1)[0][0]

def fit_screen(image):
    screen_width = 1000
    screen_height = 700
    h, w = np.shape(image)[:2]
    scale = min(screen_height/h, screen_width/w)
    newh, neww = int(scale*h), int(scale*w)

    new_image = cv2.resize(image, (neww, newh))
    return new_image
    

def recognize_faces_from_path(image_path: str, model = "hog", encoding_location: Path = DEFAULT_ENCODING_PATH):
    with encoding_location.open(mode="rb") as f:
        encodings_data = pickle.load(f)

    input_image = face_recognition.load_image_file(image_path)
    resized_image = fit_screen(input_image)
    
    input_image_face_location = face_recognition.face_locations(resized_image, model=model)
    input_image_face_encoding = face_recognition.face_encodings(resized_image, input_image_face_location)

    output = []

    if len(input_image_face_encoding) == 0:
        output.append("No_Face")
    
    for bounding_box, unknown_encoding in zip(input_image_face_location, input_image_face_encoding):
        top, right, bottom, left = bounding_box
        name = _recognize_face(unknown_encoding, encodings_data)
        if not name:
            name = "unknown"
        output.append(name)
        cv2.rectangle(resized_image, (left, top), (right, bottom), (25, 42, 133), 3)
        cv2.putText(resized_image,name,(left, max(top - 10, 25)),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2,  cv2.LINE_AA)
    cv2.imshow("image", resized_image)
    cv2.waitKey()
    return output

def recognize_faces_image(image: np.ndarray, model = "hog", encoding_location: Path = DEFAULT_ENCODING_PATH) -> list:
    with encoding_location.open(mode="rb") as f:
        encodings_data = pickle.load(f)
    bouding_boxes = face_recognition.face_locations(image, model=model)
    recognised_encodings = face_recognition.face_encodings(image, bouding_boxes)

    if(len(recognised_encodings) == 0):
        return []

    result = []

    for face_loc, encoding in zip(bouding_boxes, recognised_encodings):
        name = _recognize_face(encoding, encodings_data)
        if name == None:
            name = "Unknown"
        result.append([name, face_loc])
    
    return result


if __name__ == "__main__":
    image_path = input("Enter image path")
    recognize_faces_from_path(image_path)

## This module takes an image_path as input and gives a list of names for each face in input image.