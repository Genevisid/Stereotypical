import cv2
import numpy as np
import pandas as pd
import operator
from npwriter import f_name
import pickle
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv(f_name).values
X, Y = data[:, 1:-1], data[:, -1]
model = KNeighborsClassifier(n_neighbors = 5)
model.fit(X, Y)
filename = 'finalmodel.train'
pickle.dump(model, open(filename,'wb'))