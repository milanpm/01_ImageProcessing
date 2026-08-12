import cv2
import numpy as np


image = np.zeros((500, 500, 3), dtype=np.uint8)

points = np.array([
  [120, 180],
  [220, 100],
  [370, 150],
  [410, 250],
  [200, 390],
  [140, 350],
  [80, 260]
], dtype=np.int32)

cv2.fillPoly(image, [points], (255, 255, 255))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

contours, _ = cv2.findContours(
    gray,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

contour = contours[0]

(center_x, center_y), radius = cv2.minEnclosingCircle(contour)

center = (int(center_x), int(center_y))
radius = int(radius)

# 최소 외접원: 초록색
cv2.circle(image, center, radius, (0, 255, 0), 2)

# 원의 중심점: 빨간색
cv2.circle(image, center, 5, (0, 0, 255), -1)

cv2.imshow("Minimum Enclosing Circle", image)

cv2.waitKey(0)
cv2.destroyAllWindows()




