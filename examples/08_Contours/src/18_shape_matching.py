import cv2
import numpy as np

# 빈 이미지 생성
image = np.zeros((500, 800, 3), dtype=np.uint8)

# 비교할 도형 3개 생성
cv2.circle(image, (150, 250), 80, (255, 255, 255), -1)
cv2.circle(image, (400, 250), 60, (255, 255, 255), -1)

triangle = np.array([
    [650, 150],
    [570, 330],
    [730, 330]
], dtype=np.int32)

cv2.fillPoly(image, [triangle], (255, 255, 255))

# Grayscale 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Contour 찾기
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 왼쪽에서 오른쪽 순서로 정렬
contours = sorted(
    contours,
    key=lambda contour: cv2.boundingRect(contour)[0]
)

contour1 = contours[0]
contour2 = contours[1]
contour3 = contours[2]

# Shape Matching
score_circle_circle = cv2.matchShapes(
    contour1,
    contour2,
    cv2.CONTOURS_MATCH_I1,
    0.0
)

score_circle_triangle = cv2.matchShapes(
    contour1,
    contour3,
    cv2.CONTOURS_MATCH_I1,
    0.0
)

print(f"Circle vs Circle: {score_circle_circle:.6f}")
print(f"Circle vs Triangle: {score_circle_triangle:.6f}")

# Contour 표시
cv2.drawContours(image, [contour1], -1, (0, 255, 0), 3)
cv2.drawContours(image, [contour2], -1, (0, 255, 0), 3)
cv2.drawContours(image, [contour3], -1, (0, 0, 255), 3)

cv2.imshow("Shape Matching", image)

cv2.waitKey(0)
cv2.destroyAllWindows()