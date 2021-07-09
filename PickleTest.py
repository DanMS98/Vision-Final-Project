import cv2
import numpy as np
import pickle
import pickle
import keras
from keras.layers import Input ,Dense,Activation, Conv2D,AveragePooling2D,Flatten
from keras.models import Model
from tensorflow import ConfigProto
from tensorflow import InteractiveSession




def build_model(input_shape):
    x_input = Input(shape=input_shape, name='input')

    x = Conv2D(filters=16, kernel_size=(2, 2), strides=1, padding='valid', name='conv2')(x_input)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=2, strides=2, name='pad2')(x)

    x = Flatten()(x)

    x = Dense(units=120, name='fc_1')(x)

    x = Activation('relu', name='relu_1')(x)
    # x = Dropout(rate = 0.5)

    x = Dense(units=84, name='fc_2')(x)
    x = Activation('relu', name='relu_2')(x)
    # x = Dropout(rate = 0.5)

    outputs = Dense(units=2, name='softmax', activation='softmax')(x)

    model = Model(inputs=x_input, outputs=outputs)
    model.summary()

    return model

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


USE_TEST = False

model = build_model(input_shape=(200, 200, 3))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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
    label = "Red" if Y[Index] == 0 else "Blue"
    cv2.imshow(label, X[Index])
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break