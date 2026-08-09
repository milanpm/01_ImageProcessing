import cv2
import numpy as np

# 검은색 배경 이미지 생성
image = np.zeros((500, 500, 3), dtype=np.uint8)

# 오목한 다각형의 꼭짓점 좌표
points = np.array([
    [100, 100],
    [400, 100],
    [400, 200],
    [250, 200],
    [250, 400],
    [100, 400],
], dtype=np.int32)

# 오목한 다각형 그리기
cv2.fillPoly(image, [points], (255, 255, 255))

# 이진 이미지로 변환한 뒤 외곽 윤곽선 검출
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(
    gray,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

contour = contours[0]

# 윤곽선을 감싸는 볼록 껍질 계산
hull = cv2.convexHull(contour)

# 원본 윤곽선은 초록색, 볼록 껍질은 빨간색으로 표시
result = image.copy()
cv2.drawContours(result, [contour], -1, (0, 255, 0), 3)
cv2.drawContours(result, [hull], -1, (0, 0, 255), 3)

# 결과 이미지 출력
cv2.imshow("Original Shape and Convex Hull", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

