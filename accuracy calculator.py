import pickle
from npwriter import f_name
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
data = pd.read_csv(f_name).values
X, Y = data[:, 1:-1], data[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
model = KNeighborsClassifier(n_neighbors = 5)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
score = accuracy_score(Y_test, Y_pred)
print(score)
