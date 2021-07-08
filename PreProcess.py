import numpy as np
import os
import cv2
import random
import pickle

IMAGE_SIZE = 200
CURRENT_DIR = os.getcwd()
CATEGORIES = ["Red", "Blue"]
CURRENT_DIR = os.path.join(CURRENT_DIR, "Dataset")

training_data = []
for class_number, category in enumerate(CATEGORIES):
    path = os.path.join(CURRENT_DIR, category)
    for img in os.listdir(path):
        cur_img = cv2.imread(os.path.join(path, img))
        training_data.append([cur_img, class_number])

random.shuffle(training_data)

X = []
Y = []
for features, labels in training_data:
    X.append(features)
    Y.append(labels)

X = np.array(X).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)

X_pickle = open("Dataset/train_x.pickle", "wb")
pickle.dump(X, X_pickle)
X_pickle.close()

Y_pickle = open("Dataset/train_y.pickle", "wb")
pickle.dump(Y, Y_pickle)
Y_pickle.close()

