"""

File Name: 24_contour_solidity.py

Created: 2026.08.20

Author: Alex

Description:
    Demonstrates how to calculate contour solidity by comparing
    the contour area with its convex hull area.

"""

import cv2
import numpy as np


# Create a blank image
image = np.zeros((500, 500, 3), dtype=np.uint8)

# Create a concave polygon
points = np.array([
    [100, 100],
    [400, 100],
    [400, 200],
    [250, 200],
    [250, 400],
    [100, 400]
], dtype=np.int32)

cv2.fillPoly(image, [points], (255, 255, 255))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find contours
contours, _ = cv2.findContours(
    gray,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

contour = max(contours, key=cv2.contourArea)

# Calculate contour area
contour_area = cv2.contourArea(contour)

# Calculate convex hull
hull = cv2.convexHull(contour)

# Calculate convex hull area
hull_area = cv2.contourArea(hull)

# Calculate solidity
solidity = contour_area / hull_area

print(f"Contour Area: {contour_area:.2f}")
print(f"Convex Hull Area: {hull_area:.2f}")
print(f"Solidity: {solidity:.4f}")

# Draw contour and convex hull
cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
cv2.drawContours(image, [hull], -1, (0, 0, 255), 2)

cv2.imshow("Contour Solidity", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
