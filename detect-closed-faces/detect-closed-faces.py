# type: ignore
# from flas import Flask, request, make_response, send_file
import cv2
import matplotlib.pyplot as plt
import math
import glob


#@app.route('/detect-closed-faces', methods=['POST'])
#def calculate_closed_faces():
#    image = cv.imread()
#image = cv2.imread('data/image.jpg')

def run_image(path):
    image = cv2.imread(path)
    
    if image is None:
        print("no image found")
        print(path)
        exit()
    
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    
    
    face_center_coordinates = [[x+w/2,y+h/2] for x,y,w,h in faces]
    
    for i, face1 in enumerate(face_center_coordinates):
        for j, face2 in enumerate(face_center_coordinates):
            if i >= j:
                # already calculated
                continue
            distance = math.sqrt((face1[0] - face2[0]) ** 2 + (face1[1] - face2[1]) ** 2)
    
    for (x,y,w,h) in faces:
        image = cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 10)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show();


image_paths = glob.glob('edited/*.png')
for path in image_paths:
    run_image(path)

