import cv2
import numpy as np


def calculate_rectangularity(contour):
    contour_area = cv2.contourArea(contour)

    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    rectangle_area = width * height

    if rectangle_area == 0:
        return 0.0

    return contour_area / rectangle_area


shapes = {}

rectangle = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(rectangle, (60, 90), (240, 210), 255, -1)
shapes["Rectangle"] = rectangle

circle = np.zeros((300, 300), dtype=np.uint8)
cv2.circle(circle, (150, 150), 90, 255, -1)
shapes["Circle"] = circle

triangle = np.zeros((300, 300), dtype=np.uint8)
triangle_points = np.array(
    [[150, 50], [50, 240], [250, 240]],
    dtype=np.int32
)
cv2.fillPoly(triangle, [triangle_points], 255)
shapes["Triangle"] = triangle

result_images = []

for name, image in shapes.items():
    contours, _ = cv2.findContours(
        image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contour = max(contours, key=cv2.contourArea)
    rectangularity = calculate_rectangularity(contour)

    result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.drawContours(result, [box], 0, (0, 0, 255), 2)

    cv2.putText(
        result,
        f"{name}: {rectangularity:.4f}",
        (20, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 0, 0),
        2
    )

    print(f"{name} Rectangularity: {rectangularity:.4f}")
    result_images.append(result)

combined = np.hstack(result_images)

cv2.imshow("Contour Rectangularity", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()