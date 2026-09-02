"""
Day 45 - Extreme Points of a Contour

This example demonstrates how to detect the four extreme points
of an object's contour using OpenCV:

- Leftmost point
- Rightmost point
- Topmost point
- Bottommost point

The detected points can be used to analyze an object's position,
width, height, alignment, and region of interest.

Applications:
- Machine vision inspection
- Object alignment
- Dimension measurement
- ROI configuration
- Position and orientation analysis

Author: Alex
Project: 01_ImageProcessing
File: 32_extreme_points.py
"""

import cv2
import numpy as np


# Create a blank image
image = np.zeros((600, 800, 3), dtype=np.uint8)

# Define an irregular object
object_points = np.array(
    [
        [170, 300],
        [240, 150],
        [400, 100],
        [570, 180],
        [650, 320],
        [560, 470],
        [350, 520],
        [210, 440],
    ],
    dtype=np.int32,
)

# Draw the object
cv2.fillPoly(image, [object_points], (255, 255, 255))

# Convert to grayscale and create a binary image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find external contours
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

if not contours:
    raise RuntimeError("No contour was detected.")

# Select the largest contour
contour = max(contours, key=cv2.contourArea)

# Find the four extreme points
leftmost = tuple(map(int, contour[contour[:, :, 0].argmin()][0]))
rightmost = tuple(map(int, contour[contour[:, :, 0].argmax()][0]))
topmost = tuple(map(int, contour[contour[:, :, 1].argmin()][0]))
bottommost = tuple(map(int, contour[contour[:, :, 1].argmax()][0]))

# Calculate dimensions between the extreme coordinates
object_width = rightmost[0] - leftmost[0]
object_height = bottommost[1] - topmost[1]

# Create the result image
result = image.copy()
cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)

extreme_points = [
    ("Left", leftmost, (255, 0, 0)),
    ("Right", rightmost, (0, 0, 255)),
    ("Top", topmost, (0, 255, 255)),
    ("Bottom", bottommost, (255, 0, 255)),
]

# Draw and label each extreme point
for label, point, color in extreme_points:
    cv2.circle(result, point, 8, color, -1)
    cv2.putText(
        result,
        label,
        (point[0] + 10, point[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )

# Draw width and height measurement lines
cv2.line(
    result,
    (leftmost[0], 560),
    (rightmost[0], 560),
    (255, 255, 0),
    2,
)
cv2.line(
    result,
    (720, topmost[1]),
    (720, bottommost[1]),
    (255, 255, 0),
    2,
)

cv2.putText(
    result,
    f"Width: {object_width} px",
    (310, 590),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 0),
    2,
    cv2.LINE_AA,
)

cv2.putText(
    result,
    f"Height: {object_height} px",
    (540, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 0),
    2,
    cv2.LINE_AA,
)

# Print measurement results
print(f"Leftmost point   : {leftmost}")
print(f"Rightmost point  : {rightmost}")
print(f"Topmost point    : {topmost}")
print(f"Bottommost point : {bottommost}")
print(f"Object width     : {object_width} px")
print(f"Object height    : {object_height} px")

# Display the result
cv2.imshow("Day 45 - Extreme Points of a Contour", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
