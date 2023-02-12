from icrawler.builtin import GoogleImageCrawler
import pandas as pd
import cv2
import os
import numpy as np
columns=["name"]
f=pd.read_csv("/media/geneviptop/i am the best/top10k-spotify-artist-metadata.csv",usecols=columns)
j=0
k=0
e=0
trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
for i in list(f):
        while j <= 1065:
            m=f[i][j]
            for k in range(0,81):
                try:
                    if k<10:
                        n="/media/geneviptop/i am the best/dataset/"+m+"/"
                        t=n+"00000"+str(k)+".jpg"
                        img = cv2.imread(t)
                        grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
                        img_crop = []
                        for (x, y, w, h) in face_coordinates:
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            img_crop.append(img[y:y + h, x:x + w])
                        for counter, cropped in enumerate(img_crop):
                            cv2.imwrite("/media/geneviptop/i am the best/dataset2/"+m+"/"+"00000"+str(e)+".jpg", cropped)
                    else:
                        n="/media/geneviptop/i am the best/dataset/"+m+"/"
                        t=n+"0000"+str(k)+".jpg"
                        img = cv2.imread(t)
                        grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
                        img_crop = []
                        for (x, y, w, h) in face_coordinates:
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            img_crop.append(img[y:y + h, x:x + w])
                        for counter, cropped in enumerate(img_crop):
                            cv2.imwrite("/media/geneviptop/i am the best/dataset2/"+m+"/"+"0000"+str(e)+".jpg", cropped)
                    e=e+1
                    continue
                except:
                    continue
            j=j+1
            continue