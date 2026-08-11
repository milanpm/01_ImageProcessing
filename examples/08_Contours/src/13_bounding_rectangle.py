import cv2
import numpy as np


image = np.zeros((500, 500, 3), dtype=np.uint8)

points = np.array([
    [150, 100],
    [400, 200],
    [350, 400],
    [100, 300]
], dtype=np.int32)

cv2.fillPoly(image, [points], (255, 255, 255))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

contours, _ = cv2.findContours(
    gray,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

contour = contours[0]

# 축에 평행한 경계 사각형
x, y, width, height = cv2.boundingRect(contour)

# 도형의 회전을 고려한 최소 면적 사각형
rotated_rect = cv2.minAreaRect(contour)
box = cv2.boxPoints(rotated_rect)
box = np.int32(box)

# 축에 평행한 경계 사각형: 초록색
cv2.rectangle(
    image,
    (x, y),
    (x + width, y + height),
    (0, 255, 0),
    2
)

# 회전된 최소 면적 사각형: 빨간색
cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

cv2.imshow("Bounding Rectangles", image)

cv2.waitKey(0)
cv2.destroyAllWindows()




