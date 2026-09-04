"""
File: 33_centroid_extreme_alignment.py
Date: 2026-09-04
Author: Alex
Description:
    Demonstrates how to calculate a contour centroid using image moments,
    find the leftmost, rightmost, topmost, and bottommost contour points,
    and measure the alignment between the centroid and extreme-point midpoints.
"""

import cv2
import numpy as np
import math


# Create a blank image
image = np.zeros((500, 700, 3), dtype=np.uint8)

# Create an asymmetric object
points = np.array(
    [
        [170, 100],
        [430, 80],
        [550, 210],
        [500, 390],
        [300, 430],
        [120, 330],
        [90, 190],
    ],
    dtype=np.int32,
)

cv2.fillPoly(image, [points], (255, 255, 255))

# Convert to grayscale and create a binary image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find the largest contour
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

contour = max(contours, key=cv2.contourArea)

# Calculate the contour centroid using image moments
moments = cv2.moments(contour)

if moments["m00"] == 0:
    raise ValueError("Cannot calculate centroid: contour area is zero.")

centroid_x = int(moments["m10"] / moments["m00"])
centroid_y = int(moments["m01"] / moments["m00"])
centroid = (centroid_x, centroid_y)

# Find the four extreme points
leftmost = tuple(map(int, contour[contour[:, :, 0].argmin()][0]))
rightmost = tuple(map(int, contour[contour[:, :, 0].argmax()][0]))
topmost = tuple(map(int, contour[contour[:, :, 1].argmin()][0]))
bottommost = tuple(map(int, contour[contour[:, :, 1].argmax()][0]))

# Calculate the midpoint between opposite extreme points
horizontal_midpoint = (
    (leftmost[0] + rightmost[0]) // 2,
    (leftmost[1] + rightmost[1]) // 2,
)

vertical_midpoint = (
    (topmost[0] + bottommost[0]) // 2,
    (topmost[1] + bottommost[1]) // 2,
)

# Calculate distances from the centroid to each extreme point
def distance(point1, point2):
    return math.hypot(
        point1[0] - point2[0],
        point1[1] - point2[1],
    )


left_distance = distance(centroid, leftmost)
right_distance = distance(centroid, rightmost)
top_distance = distance(centroid, topmost)
bottom_distance = distance(centroid, bottommost)

# Calculate alignment errors
horizontal_error = distance(centroid, horizontal_midpoint)
vertical_error = distance(centroid, vertical_midpoint)

# Draw the contour
cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

# Draw lines connecting opposite extreme points
cv2.line(image, leftmost, rightmost, (255, 255, 0), 2)
cv2.line(image, topmost, bottommost, (255, 0, 255), 2)

# Draw lines from the centroid to each extreme point
for point in [leftmost, rightmost, topmost, bottommost]:
    cv2.line(image, centroid, point, (100, 100, 100), 1)

# Draw and label the extreme points
extreme_points = {
    "Left": leftmost,
    "Right": rightmost,
    "Top": topmost,
    "Bottom": bottommost,
}

for label, point in extreme_points.items():
    cv2.circle(image, point, 7, (0, 0, 255), -1)
    cv2.putText(
        image,
        label,
        (point[0] + 10, point[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
    )

# Draw the centroid and opposite-point midpoints
cv2.circle(image, centroid, 8, (255, 0, 0), -1)
cv2.circle(image, horizontal_midpoint, 6, (0, 165, 255), -1)
cv2.circle(image, vertical_midpoint, 6, (0, 165, 255), -1)

cv2.putText(
    image,
    "Centroid",
    (centroid[0] + 10, centroid[1] - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (255, 0, 0),
    2,
)

# Print measurement results
print(f"Contour area: {cv2.contourArea(contour):.2f}")
print(f"Centroid: {centroid}")
print(f"Leftmost: {leftmost}")
print(f"Rightmost: {rightmost}")
print(f"Topmost: {topmost}")
print(f"Bottommost: {bottommost}")
print()
print(f"Centroid-to-left distance: {left_distance:.2f}")
print(f"Centroid-to-right distance: {right_distance:.2f}")
print(f"Centroid-to-top distance: {top_distance:.2f}")
print(f"Centroid-to-bottom distance: {bottom_distance:.2f}")
print()
print(f"Horizontal extreme midpoint: {horizontal_midpoint}")
print(f"Vertical extreme midpoint: {vertical_midpoint}")
print(f"Horizontal alignment error: {horizontal_error:.2f}")
print(f"Vertical alignment error: {vertical_error:.2f}")

cv2.imshow("Centroid and Extreme-Point Alignment", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
