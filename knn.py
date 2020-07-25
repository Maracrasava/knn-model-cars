import pandas as pd
import sklearn
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
import numpy as np

data = pd.read_csv("car.data")

ch = preprocessing.LabelEncoder()

buying = ch.fit_transform(list(data["buying"]))
maintenance = ch.fit_transform(list(data["maint"]))
doors = ch.fit_transform(list(data["doors"]))
persons = ch.fit_transform(list(data["persons"]))
lug_boot = ch.fit_transform(list(data["lug_boot"]))
safety = ch.fit_transform(list(data["safety"]))
cl = ch.fit_transform(list(data["class"]))

predict = "class"
X = list(zip(buying, maintenance, doors, persons, lug_boot, safety))
y = list(cl)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)


"""
best = 0
for _ in range(50):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    h = KNeighborsClassifier(n_neighbors=9)
    h.fit(X_train, y_train)
    acc = h.score(X_test, y_test)
    if acc > best:
        best = acc
        print(best)
        with open("bestmodel", "wb") as file:
            pickle.dump(h, file)"""

pickle_in = open("bestmodel", "rb")
h = pickle.load(pickle_in)
prediction = h.predict(X_test)
classes = ["unacc", "acc", "good", "vgood"]

for x in range(len(prediction)):
    print(f"Prediction {classes[prediction[x]]}, Real: {classes[y_test[x]]}")



