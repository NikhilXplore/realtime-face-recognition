from pathlib import Path
from recognize_faces import recognize_faces_from_path


def performance_matrix():

    correct = 0
    count = 0
    for image_path in Path("validation").glob("*/*"):
        name = image_path.parent.name

        predicted_name = recognize_faces_from_path(str(image_path))
        
        if name == predicted_name[0]:
            if predicted_name[0] == name:
                print(name, predicted_name, "*^*")
            correct += 1
        else:
            print(name, predicted_name, "*o*")
        count += 1

    accuracy = correct/count
    print("Total: ", count, "Correct: ", correct)
    print("accuracy: ", accuracy)