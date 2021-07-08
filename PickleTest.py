import cv2
import numpy as np
import pickle

pickle_in_imgs = open("Dataset/train_x.pickle", "rb")
X = pickle.load(pickle_in_imgs)

pickle_in_lbls = open("Dataset/train_y.pickle", "rb")
Y = pickle.load(pickle_in_lbls)

while True:
    Index = np.random.randint(len(X))
    label = "Red" if Y[Index] == 0 else "Blue"
    cv2.imshow(label, X[Index])
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break