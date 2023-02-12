import cv2
import numpy as np
import pickle
import npwriter
import pandas as pd
import operator
from sklearn.neighbors import KNeighborsClassifier
f_name="final.csv"
filename = 'finalmodel.train'
mod=open(filename,'rb+')
model=pickle.load(mod)
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
X_test = []
def pred():
    cap = cv2.VideoCapture(0)
    while True:
            ret, frame = cap.read()
            cv2.imshow('Press c to capture your face', frame)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                cv2.destroyAllWindows()
                cap.release()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = classifier.detectMultiScale(gray, 1.5, 5)
                for face in faces:
                    x, y, w, h = face
                    im_face = gray[y:y + h, x:x + w]
                    im_face = cv2.resize(im_face, (100, 100))
                    X_test.append(im_face.reshape(-1))
                if len(faces) > 0:
                    response = model.predict(np.array(X_test))
                print("The name of the artist whose fanbase you probably fit into is "+response[0])
                break
def submit(X):
    name=input("Type the name of your favourite artist")
    npwriter.write(name, np.array(X))
    exec(open("model creator 2.py").read())
    print("Database updates!")
pred()
t=int(input("Wrong prediction? Enter 1 to upload your face to improve stereo-typical's predictions!"))
if(t==1):
    submit(X_test)