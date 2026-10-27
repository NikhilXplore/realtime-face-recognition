import face_recognition
import pickle
from pathlib import Path

DEFAULT_ENCODING_PATH = Path("output/encoding2.pkl")

def encode_known_faces(model: str = "hog", encodings_location: Path = DEFAULT_ENCODING_PATH) -> None:
    names =  []
    encodings = []
    count = 1
    for filepath in Path("training/my_data").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath);

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)
        print(name, count)
        count+=1

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

if __name__ == "__main__":
    encode_known_faces()