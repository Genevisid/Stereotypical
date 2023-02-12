import cv2
import numpy as np
import pandas as pd
import pickle
import npwriter
import streamlit as st
import time
from streamlit_extras.switch_page_button import switch_page
import time
file=open('1.jpg', 'wb')
m=open('finalmodel.train', 'rb')
model = pickle.load(m)
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
X_test = []
def pred():
    picture=cv2.imread('1.jpg')
    gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.5, 5)
    for face in faces:
        x, y, w, h = face
        im_face = gray[y:y + h, x:x + w]
        im_face = cv2.resize(im_face, (100, 100))
        X_test.append(im_face.reshape(-1))
        if len(faces) > 0:
            response = model.predict(np.array(X_test))
        st.write("The name of the artist whose fanbase you probably fit into is "+response[0])
        st.write("Wrong prediction? Press the submit button to upload your face to improve stereo-typical's predictions!")
def submit():
    X=[]
    img = cv2.imread('1.jpg')
    gray_face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.resize(gray_face, (100, 100))
    X.append(gray_face.reshape(-1))
    st.text_input("Type the name of your favourite artist",key="name")
    time.sleep(15)
    st.write("Loading.....")
    npwriter.write(st.session_state.name, np.array(X))
    exec(open("model creator 2.py").read())
    st.write("Database updated!")
    st.write("Returning to main menu...")
    time.sleep(2)
    switch_page("main")
butt2 = st.button("Submit your face!")
butt3 = st.button("Quit")
picture = st.camera_input("Press the button below to take a pic of your face!")
if picture:
    file.write(picture.getbuffer())
    pred()
if butt2:
    submit()
if butt3:
    switch_page("main")