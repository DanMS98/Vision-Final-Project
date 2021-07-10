import cv2
import numpy as np
import pickle
import keras
from tensorflow import ConfigProto
from tensorflow import InteractiveSession


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


USE_TEST = False

# model = build_model(input_shape=(200, 200, 3))
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

pickle_in_imgs = open("Dataset/Train/train_x.pickle", "rb")
X = pickle.load(pickle_in_imgs)
pickle_in_imgs.close()

pickle_in_lbls = open("Dataset/Train/train_y.pickle", "rb")
Y = pickle.load(pickle_in_lbls)
pickle_in_lbls.close()


model = keras.models.load_model('Dataset/Model.h5')

# print(X.shape, len(Y))

while True:
    Index = np.random.randint(len(X))
    batch = np.expand_dims(X[Index], axis=0)
    prediction = model.predict(batch)
    print(np.argmax(prediction[0]))
    # label = "Label"
    if Y[Index] == 0:
        label = "Red"
    elif Y[Index] == 1:
        label = "Blue"
    else:
        label = "Yellow"
    cv2.imshow(label, X[Index])
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break