import cv2
import numpy as np
import os
import pickle
# import pybgs as bgs
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

MAPPED_Y_OFFSET = 70
BOX_END_OFFSET = 5
BOX_START_OFFSET = 10
USE_RECOGNITION = True
GENERATE_DATASET = False
GENERATE_TEST = False
APPROXIMATE_NUMBER_OF_DATASET_SAMPLES = 1000
CURRENT_DIR = os.getcwd()
PATCH_DIM = (200, 200)
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


cam0_addr = "/home/danial/PycharmProjects/VisionCourse_kntu/Project/FirstHalf/test/output.mp4"
# cam1_addr = "/home/danial/PycharmProjects/VisionCourse_kntu/Project/FirstHalf/1/output1.mp4"
# cam2_addr = "/home/danial/PycharmProjects/VisionCourse_kntu/Project/FirstHalf/2/output2.mp4"

cam0 = cv2.VideoCapture(cam0_addr)
# cam1 = cv2.VideoCapture(cam1_addr)
# cam2 = cv2.VideoCapture(cam2_addr)

if False:
    backSub = cv2.createBackgroundSubtractorMOG2()
else:
    backSub = cv2.createBackgroundSubtractorKNN()

fieldImage = cv2.imread("2D_field.png")

generation_finished_flag = False

output_size = fieldImage.shape
n, m, _ = output_size
# print(n, m)
# field
p1 = (141, 171)
p2 = (642, 110)  # until 1004, 63
p3 = (1141, 119)
p4 = (874, 784)
#  * m/105 * n/68
# image
p1_2 = (165, 154)
p2_2 = (526, 6)
p3_2 = (888, 153)
p4_2 = (526, 696)

points1 = np.array([p1, p2, p3, p4], dtype=np.float32)
points2 = np.array([p1_2, p2_2, p3_2, p4_2], dtype=np.float32)

opening_kernel = np.ones((2, 2), np.uint8)
closing_kernel = np.ones((3, 3), np.uint8)
erode_kernel = np.ones((2, 2), np.uint8)
dataset_index_blue = 0
dataset_index_red = 0

model = build_model(input_shape=(200, 200, 3))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model = keras.models.load_model('Dataset/Model.h5')
model.load_weights("Dataset/model_weights.h5")

while True:
    fieldImage = cv2.imread("2D_field.png")
    # cv2.line(fieldImage, (0, 0), (380, 665), (100, 50, 200), 5)
    # cv2.line(fieldImage, (1049, 43), (601, 665), (100, 50, 200), 5)
    ret0, I0 = cam0.read()
    # ret1, I1 = cam1.read()
    # ret2, I2 = cam2.read()
    ret = ret0  # & ret1 & ret2

    if ret:
        # for i in range(4):
        #     cv2.circle(I0, (points1[i, 0], points1[i, 1]), 5, [0, 0, i*50], 2)

        H = cv2.getPerspectiveTransform(points1, points2)
        J = cv2.warpPerspective(I0, H, (output_size[1], output_size[0]))
        # J = cv2.GaussianBlur(J, (7, 7), 0)
        J = J[34:, :, :]
        J_copy = J.copy()

        mask = backSub.apply(J)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)
        threshold = 200
        ret, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
        # mask = cv2.erode(mask, erode_kernel)

        n, C, stats, centroids = cv2.connectedComponentsWithStats(mask)

        for i in range(1, n):
            left, top, width, height, area = stats[i][0], stats[i][1], stats[i][2], stats[i][3], stats[i][4]
            if area > 150:
                start_point = (left - BOX_START_OFFSET, top - BOX_START_OFFSET)
                end_point = (left + width + BOX_END_OFFSET, top + height + BOX_END_OFFSET)
                mappedPoint = (int(centroids[i][0]), int(centroids[i][1] + MAPPED_Y_OFFSET))
                check_vals = list([start_point[1], end_point[1], start_point[0], end_point[0]])
                patch = J_copy[start_point[1]:end_point[1], start_point[0]:end_point[0], :]


                if GENERATE_DATASET:
                    if all(i >= 0 for i in check_vals):
                        patch = cv2.resize(patch, PATCH_DIM, interpolation=cv2.INTER_AREA)
                        hsl = cv2.cvtColor(patch, cv2.COLOR_BGR2HLS)
                        hsl_mask = cv2.inRange(hsl, np.array([0, 185, 0]), np.array([255, 255, 255]))
                        # print("whites:", np.sum(hsl_mask == 255))
                        # cv2.imshow("Patch", patch)
                        # cv2.imshow("mask", hsl_mask)
                        # cv2.waitKey(0)

                        if np.sum(hsl_mask == 255) > 120 and dataset_index_red <= APPROXIMATE_NUMBER_OF_DATASET_SAMPLES:
                            print("Generating Red...")
                            subPath = '/Dataset/Test/Red/' if GENERATE_TEST else '/Dataset/Train/Red/'
                            filename = CURRENT_DIR + subPath + str(dataset_index_red) + '.png'
                            dataset_index_red = dataset_index_red + 1
                        elif np.sum(hsl_mask == 255) <= 500 and dataset_index_blue <= APPROXIMATE_NUMBER_OF_DATASET_SAMPLES:
                            print("Generating Blue...")
                            subPath = '/Dataset/Test/Blue/' if GENERATE_TEST else '/Dataset/Train/Blue/'
                            filename = CURRENT_DIR + subPath + str(dataset_index_blue) + '.png'
                            dataset_index_blue = dataset_index_blue + 1
                        else:
                            if not generation_finished_flag and\
                                    dataset_index_red > APPROXIMATE_NUMBER_OF_DATASET_SAMPLES and\
                                    dataset_index_blue > APPROXIMATE_NUMBER_OF_DATASET_SAMPLES:
                                generation_finished_flag = True
                                print("DATASET GENERATION FINISHED.")
                        cv2.imwrite(filename, patch)

                if USE_RECOGNITION:
                    if all(i >= 0 for i in check_vals):
                        patch = cv2.resize(patch, PATCH_DIM, interpolation=cv2.INTER_AREA)
                        batch = np.expand_dims(patch, axis=0)
                        prediction = model.predict(batch)
                        result = np.argmax(prediction[0])
                        player_color = [255, 0, 0] if result == 1 else [0, 0, 255]
                    else:
                        player_color = [0, 0, 255]
                else:
                    player_color = [255, 0, 255]

                cv2.circle(fieldImage, mappedPoint, 8, player_color, -1)
                cv2.circle(mask, (int(centroids[i][0]), int(centroids[i][1])), 5, player_color, 2)
                cv2.rectangle(J, start_point, end_point, [0, 255, 255], 2)

        # cv2.imshow("bgs", mask)
        cv2.imshow("Map", fieldImage)
        # cv2.imshow("Transformed Image", J)
        cv2.imshow("Original Image", I0)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break
