import cv2
import numpy as np
import time
import math
import threading
from recognize_faces import recognize_faces_image

cap = cv2.VideoCapture(0)

condition = threading.Condition()
is_processing_image = False
processed_frame = np.array(0)
quit_thread = False
Result = []
scale = 2

def face_detection():
    global is_processing_image
    global processed_frame
    global Result

    while True:
        with condition:
            condition.wait_for(lambda: is_processing_image or quit_thread)
            image = processed_frame
            
            if quit_thread:
                break
            is_processing_image = False

        small_image = cv2.resize(image, (0,0), fx=1/scale, fy=1/scale)
        rgb_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2RGB)
        result = recognize_faces_image(rgb_image)

        with condition:
            Result = result
            
        
def print_location(image):
    with condition:
        new_result = Result.copy()

    if len(new_result) == 0:
        return
    
    for item in new_result:
        face_loc = item[1]
        name = item[0]
        face_loc = [i*scale for i in face_loc]
        top, right, bottom, left = face_loc
        
        cv2.rectangle(image, (left, top), (right, bottom), (20, 255, 20), 4)
        cv2.putText(image, name, (left, max(top - 10, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


t1 = threading.Thread(target=face_detection)
t1.start()
    
previous_time = time.time()

while True:
    _, img = cap.read()
    inverted_img = cv2.flip(img, 1)

    with condition:
        if not is_processing_image:
            is_processing_image = True
            processed_frame = inverted_img.copy()
            condition.notify()
    
    
    print_location(inverted_img)


    current_time = time.time()
    delay = current_time - previous_time
    fps = math.floor(1/delay)
    previous_time = current_time
    cv2.putText(inverted_img, str(fps), (10, 50), 1, 2, (25, 42, 133))

    cv2.imshow("Window", inverted_img)
    if cv2.waitKey(1) == ord('q'):
        break

with condition:
    quit_thread = True
    condition.notify()

t1.join()
cap.release()
cv2.destroyAllWindows()