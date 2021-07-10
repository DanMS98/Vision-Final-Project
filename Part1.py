import cv2
import numpy as np
import os
import keras
from keras.layers import Input, Dense, Activation, Conv2D, AveragePooling2D, Flatten
from keras.models import Model
from tensorflow import ConfigProto
from tensorflow import InteractiveSession


class Stitcher:
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.cachedH = None

    def stitch(self, images, ratio=0.75, reprojThresh=4.0):

        (imageB, imageA) = images
        if self.cachedH is None:
            (kpsA, featuresA) = self.sift.detectAndCompute(imageA, None)
            (kpsB, featuresB) = self.sift.detectAndCompute(imageB, None)
            matches = cv2.BFMatcher().knnMatch(featuresA, featuresB, k=2)
            good_matches = []
            alpha = 0.75
            for m1, m2 in matches:
                if m1.distance < alpha * m2.distance:
                    good_matches.append(m1)

            points1 = np.array([kpsA[m.queryIdx].pt for m in good_matches], dtype=np.float32)
            points2 = np.array([kpsB[m.trainIdx].pt for m in good_matches], dtype=np.float32)
            H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
            self.cachedH = H

        result = cv2.warpPerspective(imageA, self.cachedH, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        return result


def get_result_from_model(patch, model):
    patch = cv2.resize(patch, PATCH_DIM, interpolation=cv2.INTER_AREA)
    batch = np.expand_dims(patch, axis=0)
    prediction = model.predict(batch)
    # print(prediction[0])
    result = np.argmax(prediction[0])
    if result == 0:
        color = [0, 0, 255]
    elif result == 1:
        color = [255, 0, 0]
    else:
        color = [0, 255, 255]
    return color


def build_model(input_shape):
    x_input = Input(shape=input_shape, name='input')

    x = Conv2D(filters=6, kernel_size=5, strides=1, padding='valid', name='conv1')(x_input)
    x = Activation('relu', name='relu_1')(x)
    x = AveragePooling2D(pool_size=2, strides=2, name='pad1')(x)

    x = Conv2D(filters=16, kernel_size=(2, 2), strides=1, padding='valid', name='conv2')(x_input)
    x = Activation('relu', name='relu_2')(x)
    x = AveragePooling2D(pool_size=2, strides=2, name='pad2')(x)

    x = Flatten()(x)

    x = Dense(units=120, name='fc_1')(x)

    x = Activation('relu', name='relu_3')(x)
    # x = Dropout(rate = 0.3)

    x = Dense(units=84, name='fc_2')(x)
    x = Activation('relu', name='relu_4')(x)
    # x = Dropout(rate = 0.3)

    outputs = Dense(units=3, name='softmax', activation='softmax')(x)

    model = Model(inputs=x_input, outputs=outputs)
    model.summary()

    return model


def stitch_frames(I0, I1, I2):
    stitcher01 = Stitcher()
    stitcher12 = Stitcher()
    stitcher = Stitcher()

    frames_01 = []
    frames_12 = []
    frames_01.append(I0)
    frames_01.append(I1)

    frames_12.append(I1)
    frames_12.append(I2)

    stitch01 = stitcher01.stitch(frames_01)
    stitch01 = cv2.resize(stitch01, (I0.shape[1], I0.shape[0]), interpolation=cv2.INTER_AREA)

    stitch12 = stitcher12.stitch(frames_12)
    stitch12 = cv2.resize(stitch12, (I0.shape[1], I0.shape[0]), interpolation=cv2.INTER_AREA)

    last_stitch = [stitch01, stitch12]

    stitch = stitcher.stitch(last_stitch)
    stitch = cv2.resize(stitch, (I0.shape[1], I0.shape[0]), interpolation=cv2.INTER_AREA)

    return stitch


USE_STITCHING = False
USE_RECOGNITION = True
USE_OPTICAL_FLOW = True
GENERATE_DATASET = False
GENERATE_TEST = False
USE_MODEL_WEIGHTS = False
USE_MOG2 = False
MAPPED_Y_OFFSET = 70
BOX_END_OFFSET = 5
BOX_START_OFFSET = 10
OPTICALFLOW_FRAME_COUNTER_LIMIT = 8
BASE_AREA_LIMIT = 50
APPROXIMATE_NUMBER_OF_DATASET_SAMPLES = 1300
CURRENT_DIR = os.getcwd()
PATCH_DIM = (200, 200)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

cam1_addr = "/home/danial/PycharmProjects/VisionCourse_kntu/Project/FirstHalf/main/output.mp4"
cam0_addr = "/home/danial/PycharmProjects/VisionCourse_kntu/Project/FirstHalf/cam0/output0.mp4"
cam2_addr = "/home/danial/PycharmProjects/VisionCourse_kntu/Project/FirstHalf/cam2/output2.mp4"

fieldImage = cv2.imread("2D_field.png")

cam1 = cv2.VideoCapture(cam1_addr)
cam0 = cv2.VideoCapture(cam0_addr)
cam2 = cv2.VideoCapture(cam2_addr)

if USE_MOG2:
    backSub = cv2.createBackgroundSubtractorMOG2()
else:
    backSub = cv2.createBackgroundSubtractorKNN()


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
frame_counter = OPTICALFLOW_FRAME_COUNTER_LIMIT
lk_params = dict(winSize=(30, 30), maxLevel=1,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.8))

if USE_RECOGNITION:
    if USE_MODEL_WEIGHTS:
        model = build_model(input_shape=(200, 200, 3))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.load_weights("Dataset/model_weights.h5")
    else:
        model = keras.models.load_model('Dataset/Model.h5')

_, old_frame1 = cam1.read()
_, old_frame0 = cam0.read()
_, old_frame2 = cam2.read()
old_points_for_flow = []
while True:
    fieldImage = cv2.imread("2D_field.png")
    ret1, I1 = cam1.read()
    ret0, I0 = cam0.read()
    ret2, I2 = cam2.read()
    ret = ret0 & ret1 & ret2
    if ret:
        if USE_STITCHING:
            I1 = stitch_frames(I0, I1, I2)

        # for i in range(4):
        #     cv2.circle(I0, (points1[i, 0], points1[i, 1]), 5, [0, 0, i*50], 2)

        H = cv2.getPerspectiveTransform(points1, points2)
        J = cv2.warpPerspective(I1, H, (output_size[1], output_size[0]))
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
        if USE_OPTICAL_FLOW and frame_counter < OPTICALFLOW_FRAME_COUNTER_LIMIT:

            old_points = [p[0] for p in old_points_for_flow]
            colors = [p[1] for p in old_points_for_flow]
            frame_gray = cv2.cvtColor(J_copy, cv2.COLOR_BGR2GRAY)
            old_gray = cv2.cvtColor(old_frame1, cv2.COLOR_BGR2GRAY)
            old_pts = np.array([old_points], dtype="float32").reshape(-1, 1, 2)
            new_points, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_pts,
                                                           None, **lk_params)

            frame_counter = frame_counter + 1
            for i, point in enumerate(new_points):
                # print(point[0])
                cv2.circle(fieldImage, (point[0][0], point[0][1]), 8, colors[i], -1)
                # cv2.circle(mask, point, 5, player_color, 2)
                cv2.circle(J, (point[0][0], point[0][1]), 8, colors[i], -1)

        else:
            old_frame1 = J_copy
            frame_counter = 0
            old_points_for_flow = []
            for i in range(1, n):
                left, top, width, height, area = stats[i][0], stats[i][1], stats[i][2], stats[i][3], stats[i][4]
                # if top < I1.shape[0]//4:
                #     area_limit = BASE_AREA_LIMIT * 4
                # if I1.shape[0]//4 < top < I1.shape[0]//2:
                #     area_limit = BASE_AREA_LIMIT * 3
                # if I1.shape[0]//2 < top < 2*I1.shape[0]//3:
                #     area_limit = BASE_AREA_LIMIT * 2
                # if 2*I1.shape[0]//3 < top:
                #     area_limit = BASE_AREA_LIMIT * 1
                if top > 500:
                    area_limit = BASE_AREA_LIMIT
                else:
                    area_limit = BASE_AREA_LIMIT * 3


                if area > area_limit:
                    start_point = (left - BOX_START_OFFSET, top - BOX_START_OFFSET)
                    end_point = (left + width + BOX_END_OFFSET, top + height + BOX_END_OFFSET)
                    mapped_point_offset = (int(centroids[i][0]), int(centroids[i][1] + MAPPED_Y_OFFSET))
                    # old_points_for_flow.append(mapped_point_offset)
                    check_vals = list([start_point[1], end_point[1], start_point[0], end_point[0]])
                    patch = J_copy[start_point[1]:end_point[1], start_point[0]:end_point[0], :]

                    if GENERATE_DATASET:
                        if all(i >= 0 for i in check_vals):
                            patch = cv2.resize(patch, PATCH_DIM, interpolation=cv2.INTER_AREA)
                            hsl = cv2.cvtColor(patch, cv2.COLOR_BGR2HLS)
                            hsl_mask = cv2.inRange(hsl, np.array([0, 185, 0]), np.array([255, 255, 255]))
                            # print(area_limit)
                            # print("whites:", np.sum(hsl_mask == 255))
                            # cv2.imshow("Patch", patch)
                            # cv2.imshow("mask", hsl_mask)
                            # cv2.waitKey(0)

                            if np.sum(hsl_mask == 255) > 500 and dataset_index_red <= APPROXIMATE_NUMBER_OF_DATASET_SAMPLES:
                                print("Generating Red...")
                                subPath = '/Dataset/Test/Red/' if GENERATE_TEST else '/Dataset/Train/Red/'
                                filename = CURRENT_DIR + subPath + str(dataset_index_red) + '.png'
                                dataset_index_red = dataset_index_red + 1

                            elif np.sum(hsl_mask == 255) <= 500 and\
                                    dataset_index_blue <= APPROXIMATE_NUMBER_OF_DATASET_SAMPLES:
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
                            player_color = get_result_from_model(patch, model)
                        else:
                            player_color = [0, 0, 255]
                    else:
                        player_color = [0, 255, 255]

                    old_points_for_flow.append([mapped_point_offset, player_color])
                    cv2.circle(fieldImage, mapped_point_offset, 8, player_color, -1)
                    cv2.circle(mask, (int(centroids[i][0]), int(centroids[i][1])), 5, player_color, 2)
                    cv2.rectangle(J, start_point, end_point, [0, 255, 255], 2)

        cv2.imshow("bgs", mask)
        cv2.imshow("Transformed Image", J)
        cv2.imshow("Original Image1", I1)
        cv2.imshow("Map", fieldImage)
        # cv2.imshow("Original Image0", I0)
        # cv2.imshow("Original Image2", I2)
        # cv2.imshow("stitch01 ", stitch01)
        # cv2.imshow("stitch12 ", stitch)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break
