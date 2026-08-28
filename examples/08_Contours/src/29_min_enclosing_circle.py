"""

File Name: 29_min_enclosing_circle.py

Created: 2026.08.28

Author: Alex

Description:
    Demonstrates how to calculate and visualize the minimum
    enclosing circle of a contour using cv2.minEnclosingCircle().

"""

import cv2
import numpy as np


def analyze_shape(name, points):
    """Draw a shape and calculate its minimum enclosing circle."""
    image = np.zeros((400, 400, 3), dtype=np.uint8)

    contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(image, [contour], (255, 255, 255))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(
        gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    largest_contour = max(contours, key=cv2.contourArea)

    (center_x, center_y), radius = cv2.minEnclosingCircle(
        largest_contour
    )

    center = (int(round(center_x)), int(round(center_y)))
    radius_int = int(round(radius))

    cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)
    cv2.circle(image, center, radius_int, (0, 0, 255), 2)
    cv2.circle(image, center, 4, (255, 0, 0), -1)

    cv2.putText(
        image,
        f"Center: ({center_x:.1f}, {center_y:.1f})",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"Radius: {radius:.2f}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )

    print(
        f"{name}: center=({center_x:.2f}, {center_y:.2f}), "
        f"radius={radius:.2f}, diameter={2 * radius:.2f}"
    )

    cv2.imshow(name, image)


shapes = {
    "Rectangle": [(100, 120), (300, 120), (300, 280), (100, 280)],
    "Triangle": [(200, 70), (330, 300), (70, 300)],
    "Irregular Polygon": [
        (90, 160),
        (160, 70),
        (280, 100),
        (330, 220),
        (230, 320),
        (100, 280),
    ],
}

for shape_name, shape_points in shapes.items():
    analyze_shape(shape_name, shape_points)

print("\nGreen: contour")
print("Red: minimum enclosing circle")
print("Blue: circle center")
print("Press any key to close the windows.")

cv2.waitKey(0)
cv2.destroyAllWindows()
