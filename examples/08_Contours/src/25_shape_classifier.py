import cv2
import numpy as np

# 빈 이미지 생성
image = np.zeros((500, 800, 3), dtype=np.uint8)

# Circle
cv2.circle(image, (150, 250), 80, (255, 255, 255), -1)

# Rectangle
cv2.rectangle(image, (300, 170), (500, 330), (255, 255, 255), -1)

# Triangle
triangle = np.array([
    [650, 150],
    [570, 330],
    [730, 330]
], dtype=np.int32)

cv2.fillPoly(image, [triangle], (255, 255, 255))

# Grayscale 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Binary 이미지 생성
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Contour 검출
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print("Number of contours:", len(contours))

for contour in contours:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        continue

    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    x, y, w, h = cv2.boundingRect(contour)
    rect_area = w * h

    if rect_area == 0:
        continue

    rectangularity = area / rect_area
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    if hull_area == 0:
        continue

    solidity = area / hull_area
    
    if circularity > 0.85:
        shape = "Circle"
    elif rectangularity > 0.90:
        shape = "Rectangle"
    else:
        shape = "Triangle"
        
    moments = cv2.moments(contour)

    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        cv2.putText(
            image,
            shape,
            (cx - 40, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    print(
        f"Shape: {shape}, "
        f"Circularity: {circularity:.4f}, "
        f"Rectangularity: {rectangularity:.4f}, "
        f"Solidity: {solidity:.4f}"
    )

cv2.imshow("Shape Classifier", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
