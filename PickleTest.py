import cv2
import numpy as np
import pickle

pickle_in_imgs = open("Dataset/imgs.pickle", "rb")
X = pickle.load(pickle_in_imgs)

pickle_in_lbls = open("Dataset/lables.pickle", "rb")
Y = pickle.load(pickle_in_lbls)

Index = np.random.randint(len(X))

label = "Red" if Y[Index] == 0 else "Blue"
cv2.imshow(label, X[Index])
cv2.waitKey(0)
