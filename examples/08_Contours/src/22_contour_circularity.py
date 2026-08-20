import cv2
import numpy as np


def calculate_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return 0.0

    return 4 * np.pi * area / (perimeter ** 2)
  
  
 # 비교용 흑백 이미지 생성
image = np.zeros((400, 900), dtype=np.uint8)

# 원
cv2.circle(image, (150, 200), 100, 255, -1)

# 타원
cv2.ellipse(image, (450, 200), (120, 70), 0, 0, 360, 255, -1)

# 사각형
cv2.rectangle(image, (680, 100), (860, 300), 255, -1)


# 윤곽선 검출
contours, _ = cv2.findContours(
    image,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 왼쪽에서 오른쪽 순서로 정렬
contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])

shape_names = ["Circle", "Ellipse", "Rectangle"]


# 결과 표시용 컬러 이미지
result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

for contour, name in zip(contours, shape_names):
    circularity = calculate_circularity(contour)
    x, y, w, h = cv2.boundingRect(contour)

    # 윤곽선 표시
    cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)

    # 도형 이름과 원형도 표시
    label = f"{name}: {circularity:.4f}"
    cv2.putText(
        result,
        label,
        (x, y - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2
    )

    print(
        f"{name:9s} | "
        f"Area: {cv2.contourArea(contour):8.1f} | "
        f"Perimeter: {cv2.arcLength(contour, True):7.1f} | "
        f"Circularity: {circularity:.4f}"
    )

cv2.imshow("Contour Circularity", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

 
  
  
  
  