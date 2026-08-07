import cv2
import numpy as np


image = np.zeros((500, 500), dtype=np.uint8)

# 바깥쪽 흰색 사각형
cv2.rectangle(image, (50, 50), (450, 450), 255, -1)

# 내부의 검은색 구멍
cv2.rectangle(image, (150, 150), (350, 350), 0, -1)

# 검은색 구멍 안의 흰색 객체
cv2.rectangle(image, (220, 220), (280, 280), 255, -1)

contours, hierarchy = cv2.findContours(
    image,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)

print("윤곽선 개수:", len(contours))
print("Hierarchy:")
print(hierarchy)

# 윤곽선을 그릴 컬러 이미지 생성
result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

for index, contour in enumerate(contours):
    parent = hierarchy[0][index][3]

    # 부모가 있는 윤곽선만 그리기
    if parent != -1:
        cv2.drawContours(
            result,
            contours,
            index,
            (0, 0, 255),
            3
        )

        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(
            result,
            f"Child {index}",
            (x + 10, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

cv2.imshow("Contour Hierarchy", result)
cv2.waitKey(0)
cv2.destroyAllWindows()