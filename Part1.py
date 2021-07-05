import cv2
import numpy as np
# import pybgs as bgs


cam0_addr = "/home/danial/PycharmProjects/VisionCourse_kntu/Project/FirstHalf/main/output.mp4"
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
output_size = fieldImage.shape
n, m, _ = output_size
print(n, m)
p1 = (0, 153)
p2 = (642, 113)
p3 = (1141, 120)
p4 = (874, 784)
#  * m/105 * n/68
p1_2 = (0, 0)
p2_2 = (52.5 * m/105, 0 * n/68 )
p3_2 = (88.6 * m/105, 13.9 * n/68 )
p4_2 = (52.5 * m/105, 68 * n/68 )

points1 = np.array([p1, p2, p3, p4], dtype=np.float32)
points2 = np.array([p1_2, p2_2, p3_2, p4_2], dtype=np.float32)

opening_kernel = np.ones((4, 4), np.uint8)
closing_kernel = np.ones((3, 3), np.uint8)
erode_kernel = np.ones((2, 2), np.uint8)

while True:
    fieldImage = cv2.imread("2D_field.png")
    ret0, I0 = cam0.read()
    # ret1, I1 = cam1.read()
    # ret2, I2 = cam2.read()
    ret = ret0  # & ret1 & ret2

    if ret:
        # for i in range(4):
        #     cv2.circle(I0, (points1[i, 0], points1[i, 1]), 5, [0, 0, i*50], 2)

        H = cv2.getPerspectiveTransform(points1, points2)
        J = cv2.warpPerspective(I0, H, (output_size[1], output_size[0]))
        J = cv2.GaussianBlur(J, (7, 7), 0)
        J = J[35:, :, ]

        mask = backSub.apply(J)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)
        threshold = 200
        ret, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
        # mask = cv2.erode(mask, erode_kernel)

        n, C, stats, centroids = cv2.connectedComponentsWithStats(mask)

        for i in range(1, n):
            left, top, width, height, area = stats[i][0], stats[i][1], stats[i][2], stats[i][3], stats[i][4]
            if area > 130:
                start_point = (left, top)
                end_point = (left + width, top + height)
                cv2.circle(fieldImage, (int(centroids[i][0]), int(centroids[i][1])), 8, [0, 0, 255], -1)
                cv2.circle(mask, (int(centroids[i][0]), int(centroids[i][1])), 5, [0, 0, 255], 2)
                cv2.rectangle(J, start_point, end_point, [0, 255, 255], 2)

        cv2.imshow("bgs", mask)
        cv2.imshow("result", fieldImage)
        cv2.imshow("originalImage", J)
        # cv2.imshow("rawImage", I0)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
