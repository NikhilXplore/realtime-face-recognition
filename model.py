from encoding import encode_known_faces
from recognize_faces import recognize_faces_from_path
from pathlib import Path

for file_path in Path("archive").glob("*"):
    recognize_faces_from_path(str(file_path))