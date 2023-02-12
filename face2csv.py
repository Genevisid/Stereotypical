import cv2
import numpy as np
import npwriter
import pandas as pd
f_name='final.csv'
classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
f_list = []
l=100
columns=["name"]
f=pd.read_csv("top10k-spotify-artist-metadata.csv",usecols=columns)
df = pd.read_csv(f_name, index_col = 0)
for i in list(f):
    for j in range(725,1066):
        m=f[i][j]
        for k in range(0,65):
            try:
                if k < 10:
                    n="/home/geneviptop/PycharmProjects/krs/dataset2/"+m+"/"
                    t=n+"00000"+str(l)+".jpg"
                    l=l+1
                else:
                    n="/home/geneviptop/PycharmProjects/krs/dataset2/"+m+"/"
                    t=n+"0000"+str(k)+".jpg"
                img = cv2.imread(t)
                gray_face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_face = cv2.resize(gray_face, (100, 100))
                f_list.append(gray_face.reshape(-1))
            except:
                continue
        latest = pd.DataFrame(np.array(f_list), columns=map(str, range(10000)))
        latest["name"] = m
        df = pd.concat((df, latest), ignore_index=True, sort=False)
        df.to_csv(f_name)
        f_list = []
        l=100
        continue