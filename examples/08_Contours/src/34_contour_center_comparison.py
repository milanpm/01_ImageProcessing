"""
File: 34_contour_center_comparison.py
Date: 2026-09-07
Author: Alex
Description:
    Compares three methods for estimating an object's center:
    the contour centroid, bounding-box center, and minimum-enclosing-circle
    center. The example visualizes and measures the differences between
    these centers for an asymmetric contour.
"""

import math

import cv2
import numpy as np


def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.hypot(
        point1[0] - point2[0],
        point1[1] - point2[1],
    )


# Create a blank image
image = np.zeros((600, 800, 3), dtype=np.uint8)

# Define an asymmetric object
points = np.array(
    [
        [160, 100],
        [480, 80],
        [650, 220],
        [510, 290],
        [610, 470],
        [330, 520],
        [120, 400],
        [80, 210],
    ],
    dtype=np.int32,
)

cv2.fillPoly(image, [points], (255, 255, 255))

# Convert the object to a binary image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find the largest external contour
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

if not contours:
    raise ValueError("No contours were found.")

contour = max(contours, key=cv2.contourArea)

# ------------------------------------------------------------
# 1. Contour centroid using image moments
# ------------------------------------------------------------
moments = cv2.moments(contour)

if moments["m00"] == 0:
    raise ValueError("Cannot calculate centroid: contour area is zero.")

centroid = (
    moments["m10"] / moments["m00"],
    moments["m01"] / moments["m00"],
)

# ------------------------------------------------------------
# 2. Axis-aligned bounding-box center
# ------------------------------------------------------------
x, y, width, height = cv2.boundingRect(contour)

bounding_box_center = (
    x + width / 2.0,
    y + height / 2.0,
)

# ------------------------------------------------------------
# 3. Minimum-enclosing-circle center
# ------------------------------------------------------------
enclosing_circle_center, radius = cv2.minEnclosingCircle(contour)

# Convert floating-point centers for drawing
centroid_draw = tuple(map(round, centroid))
bounding_box_center_draw = tuple(map(round, bounding_box_center))
enclosing_circle_center_draw = tuple(
    map(round, enclosing_circle_center)
)

# Calculate pairwise center distances
centroid_to_box = calculate_distance(
    centroid,
    bounding_box_center,
)

centroid_to_circle = calculate_distance(
    centroid,
    enclosing_circle_center,
)

box_to_circle = calculate_distance(
    bounding_box_center,
    enclosing_circle_center,
)

# Draw the contour
cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)

# Draw the bounding box
cv2.rectangle(
    image,
    (x, y),
    (x + width, y + height),
    (255, 255, 0),
    2,
)

# Draw the minimum enclosing circle
cv2.circle(
    image,
    tuple(map(round, enclosing_circle_center)),
    round(radius),
    (255, 0, 255),
    2,
)

# Connect the three centers
cv2.line(
    image,
    centroid_draw,
    bounding_box_center_draw,
    (120, 120, 120),
    1,
)

cv2.line(
    image,
    centroid_draw,
    enclosing_circle_center_draw,
    (120, 120, 120),
    1,
)

cv2.line(
    image,
    bounding_box_center_draw,
    enclosing_circle_center_draw,
    (120, 120, 120),
    1,
)

# Define center labels and colors
center_information = [
    (
        "Centroid",
        centroid_draw,
        (0, 0, 255),
        (-120, -20),
    ),
    (
        "Box Center",
        bounding_box_center_draw,
        (255, 0, 0),
        (15, 30),
    ),
    (
        "Circle Center",
        enclosing_circle_center_draw,
        (0, 165, 255),
        (15, -20),
    ),
]

# Draw and label each center
for label, center, color, label_offset in center_information:
    cv2.circle(image, center, 8, color, -1)

    label_position = (
        center[0] + label_offset[0],
        center[1] + label_offset[1],
    )

    cv2.putText(
        image,
        label,
        label_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )

# Print analysis results
print(f"Contour area: {cv2.contourArea(contour):.2f}")
print()

print(
    "Contour centroid: "
    f"({centroid[0]:.2f}, {centroid[1]:.2f})"
)

print(
    "Bounding-box center: "
    f"({bounding_box_center[0]:.2f}, "
    f"{bounding_box_center[1]:.2f})"
)

print(
    "Enclosing-circle center: "
    f"({enclosing_circle_center[0]:.2f}, "
    f"{enclosing_circle_center[1]:.2f})"
)

print()
print(f"Bounding box: x={x}, y={y}, width={width}, height={height}")
print(f"Enclosing-circle radius: {radius:.2f}")

print()
print(f"Centroid to box-center distance: {centroid_to_box:.2f}")
print(
    "Centroid to circle-center distance: "
    f"{centroid_to_circle:.2f}"
)
print(f"Box-center to circle-center distance: {box_to_circle:.2f}")

# Show the result
cv2.imshow("Contour Center Comparison", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
