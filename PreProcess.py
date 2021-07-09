import numpy as np
import os
import cv2
import random
import pickle

IMAGE_SIZE = 200
TRAIN_DATASET = True
CURRENT_DIR = os.getcwd()
CATEGORIES = ["Red", "Blue"]
CURRENT_DIR = os.path.join(CURRENT_DIR, "Dataset")
sub_dir = "Train" if TRAIN_DATASET else "Test"
CURRENT_DIR = os.path.join(CURRENT_DIR, sub_dir)
# print(CURRENT_DIR)
# exit()
training_data = []
test_data = []
for class_number, category in enumerate(CATEGORIES):
    print(class_number, str(category))
    path = os.path.join(CURRENT_DIR, category)
    for img in os.listdir(path):
        cur_img = cv2.imread(os.path.join(path, img))
        # cv2.imshow("temp", cur_img)
        # cv2.waitKey(1)
        if TRAIN_DATASET:
            training_data.append([cur_img, class_number])
        else:
            test_data.append([cur_img, class_number])

if TRAIN_DATASET:
    random.shuffle(training_data)
    X = []
    Y = []
    for features, labels in training_data:
        X.append(features)
        Y.append(labels)
    X = np.array(X).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)

    X_pickle = open("Dataset/Train/train_x.pickle", "wb")
    pickle.dump(X, X_pickle)
    X_pickle.close()

    Y_pickle = open("Dataset/Train/train_y.pickle", "wb")
    pickle.dump(Y, Y_pickle)
    Y_pickle.close()
else:
    random.shuffle(test_data)
    X = []
    Y = []
    for features, labels in test_data:
        X.append(features)
        Y.append(labels)
    X = np.array(X).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)

    X_pickle = open("Dataset/Test/test_x.pickle", "wb")
    pickle.dump(X, X_pickle)
    X_pickle.close()

    Y_pickle = open("Dataset/Test/test_y.pickle", "wb")
    pickle.dump(Y, Y_pickle)
    Y_pickle.close()





