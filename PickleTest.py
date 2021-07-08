import cv2
import numpy as np
import pickle

Index = np.random.randint(200)

pickle_in_imgs = open("imgs.pickle", "rb")
X = pickle.load(pickle_in_imgs)

pickle_in_lbls = open("lables.pickle", "rb")
Y = pickle.load(pickle_in_lbls)

label = "Red" if Y[Index] == 0 else "Blue"
cv2.imshow(label, X[Index])
cv2.waitKey(0)