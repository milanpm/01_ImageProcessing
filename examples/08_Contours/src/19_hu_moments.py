import cv2
import numpy as np

# 빈 이미지 생성
image = np.zeros((500, 800, 3), dtype=np.uint8)

# 비교할 도형 생성
cv2.circle(image, (200, 250), 80, (255, 255, 255), -1)

triangle = np.array([
    [550, 150],
    [450, 350],
    [650, 350]
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

circle_contour = contours[0]
triangle_contour = contours[1]

# Moments 계산
circle_moments = cv2.moments(circle_contour)
triangle_moments = cv2.moments(triangle_contour)

# Hu Moments 계산
circle_hu = cv2.HuMoments(circle_moments)
triangle_hu = cv2.HuMoments(triangle_moments)

print("Circle Hu Moments")
for i, value in enumerate(circle_hu):
    print(f"Hu[{i + 1}] = {value[0]:.10e}")

print()

print("Triangle Hu Moments")
for i, value in enumerate(triangle_hu):
    print(f"Hu[{i + 1}] = {value[0]:.10e}")

# Contour 표시
cv2.drawContours(image, [circle_contour], -1, (0, 255, 0), 3)
cv2.drawContours(image, [triangle_contour], -1, (0, 0, 255), 3)

cv2.imshow("Hu Moments", image)

cv2.waitKey(0)
cv2.destroyAllWindows()