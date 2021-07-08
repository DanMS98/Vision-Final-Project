import cv2
import numpy as np
import os
import pickle
# import pybgs as bgs

MAPPED_Y_OFFSET = 70
BOX_END_OFFSET = 5
BOX_START_OFFSET = 10

GENERATE_DATASET = True
APPROXIMATE_NUMBER_OF_SAMPLES = 700

generation_finished_flag = False
cam0_addr = "/home/danial/PycharmProjects/VisionCourse_kntu/Project/FirstHalf/test/output.mp4"
# cam1_addr = "/home/danial/PycharmProjects/VisionCourse_kntu/Project/FirstHalf/1/output1.mp4"
# cam2_addr = "/home/danial/PycharmProjects/VisionCourse_kntu/Project/FirstHalf/2/output2.mp4"

cam0 = cv2.VideoCapture(cam0_addr)
# cam1 = cv2.VideoCapture(cam1_addr)
# cam2 = cv2.VideoCapture(cam2_addr)

# output_size = (n, m)
if False:
    backSub = cv2.createBackgroundSubtractorMOG2()
else:
    backSub = cv2.createBackgroundSubtractorKNN()
# background_subtr_method = bgs.SuBSENSE()

fieldImage = cv2.imread("2D_field.png")

CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)

PATCH_DIM = (200, 200)
output_size = fieldImage.shape
n, m, _ = output_size
print(n, m)
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
                if GENERATE_DATASET:
                    check_vals = list([start_point[1], end_point[1], start_point[0], end_point[0]])
                    if all(i >= 0 for i in check_vals):
                        patch = J_copy[start_point[1]:end_point[1], start_point[0]:end_point[0], :]
                        patch = cv2.resize(patch, PATCH_DIM, interpolation=cv2.INTER_AREA)


                        hsl = cv2.cvtColor(patch, cv2.COLOR_BGR2HLS)
                        hsl_mask = cv2.inRange(hsl, np.array([0, 185, 0]), np.array([255, 255, 255]))
                        # print("whites:", np.sum(hsl_mask == 255))

                        if np.sum(hsl_mask == 255) > 500 and dataset_index_red <= APPROXIMATE_NUMBER_OF_SAMPLES:
                            print("Generating Red...")
                            filename = CURRENT_DIR + '/Dataset/Red/' + str(dataset_index_red) + '.png'
                            dataset_index_red = dataset_index_red + 1
                        elif np.sum(hsl_mask == 255) <= 500 and dataset_index_blue <= APPROXIMATE_NUMBER_OF_SAMPLES:
                            print("Generating Blue...")
                            filename = CURRENT_DIR + '/Dataset/Blue/' + str(dataset_index_blue) + '.png'
                            dataset_index_blue = dataset_index_blue + 1
                        else:
                            if not generation_finished_flag and\
                                    dataset_index_red > APPROXIMATE_NUMBER_OF_SAMPLES and\
                                    dataset_index_blue > APPROXIMATE_NUMBER_OF_SAMPLES:
                                generation_finished_flag = True
                                print("DATASET GENERATION FINISHED.")

                        cv2.imwrite(filename, patch)

                cv2.circle(fieldImage, mappedPoint, 8, [0, 0, 255], -1)
                cv2.circle(mask, (int(centroids[i][0]), int(centroids[i][1])), 5, [0, 0, 255], 2)
                cv2.rectangle(J, start_point, end_point, [0, 255, 255], 2)

        # cv2.imshow("bgs", mask)
        cv2.imshow("FieldImage", fieldImage)
        cv2.imshow("TransformedImage", J)
        cv2.imshow("originalImage", I0)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break
