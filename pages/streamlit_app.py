import cv2
import numpy as np
import pandas as pd
import pickle
import npwriter
import streamlit as st
import time
from streamlit_extras.switch_page_button import switch_page
import time
file=open('1.jpg', 'wb+')
m=open('/home/geneviptop/Videos/finalmodel.train', 'rb')
model = pickle.load(m)
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt_tree.xml")
X_test = []
def pred():
    for i in range(10):
        img=cv2.imread('1.jpg')
        grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
        img_crop = []
        for (x, y, w, h) in face_coordinates:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            img_crop.append(img[y:y + h, x:x + w])
        for counter, cropped in enumerate(img_crop):
            cv2.imwrite('3.jpg',cropped)
        img2 = cv2.imread('3.jpg')
        im_face=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        im_face = cv2.resize(im_face, (100, 100))
        X_test.append(im_face.reshape(-1))
    response = model.predict(np.array(X_test))
    st.write("The name of the artist whose fanbase you probably fit into is "+response[0])
    st.write("Wrong prediction? Press the submit button to upload your face to improve stereo-typical's predictions!")
def submit():
    X=[]
    img = cv2.imread('3.jpg')
    grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    i_face=grayscaled_img
    i_face = cv2.resize(i_face, (100, 100))
    for i in range(30):
        X.append(i_face.reshape(-1))
    name=input("Type the name of your favourite artist ")
    st.write("Loading.....")
    npwriter.write(name, np.array(X))
    exec(open("modelcreator2.py").read())
    st.write("Database updated!")
    st.write("Returning to main menu...")
    time.sleep(5)
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
