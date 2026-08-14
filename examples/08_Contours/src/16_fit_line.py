import cv2
import numpy as np

# 검은색 캔버스 생성
image = np.zeros((500, 500, 3), dtype=np.uint8)

# 불규칙한 흰색 객체 생성
points = np.array([
    [120, 380],
    [180, 300],
    [230, 250],
    [290, 190],
    [350, 120],
    [390, 160],
    [330, 230],
    [270, 290],
    [210, 350],
    [150, 410]
], dtype=np.int32)

cv2.fillPoly(image, [points], (255, 255, 255))


# 그레이스케일 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 윤곽선 검출
contours, _ = cv2.findContours(
    gray,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE
)

# 가장 큰 윤곽선 선택
contour = max(contours, key=cv2.contourArea)


# 윤곽선에 가장 잘 맞는 직선 계산
vx, vy, x, y = cv2.fitLine(
    contour,
    cv2.DIST_L2,
    0,
    0.01,
    0.01
)

# 이미지 너비
height, width = image.shape[:2]

# 직선의 양쪽 끝점 계산
left_y = int((-x * vy / vx) + y)
right_y = int(((width - 1 - x) * vy / vx) + y)

start_point = (0, left_y)
end_point = (width - 1, right_y)


# 피팅된 직선 표시
cv2.line(
    image,
    start_point,
    end_point,
    (0, 255, 0),
    2
)


# 결과 출력
cv2.imshow("Fit Line", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


