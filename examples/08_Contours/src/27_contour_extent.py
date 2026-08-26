"""
File Name: 27_contour_extent.py

Created: 2026.08.26

Author: Alex

Description: Demonstrates how to calculate the extent of a contour
             using its contour area and bounding rectangle area in OpenCV.
"""

import cv2
import numpy as np


# Create a black image
image = np.zeros((500, 500, 3), dtype=np.uint8)

# Draw a filled rectangle
cv2.rectangle(image, (100, 150), (400, 300), (255, 255, 255), -1)
#cv2.circle(image, (250, 250), 100, (255, 255, 255), -1)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply threshold
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Get the largest contour
contour = max(contours, key=cv2.contourArea)

# Calculate contour area
contour_area = cv2.contourArea(contour)

# Get bounding rectangle
x, y, w, h = cv2.boundingRect(contour)

# Calculate bounding rectangle area
bounding_area = w * h

# Calculate extent
extent = contour_area / bounding_area

print(f"Contour Area: {contour_area:.2f}")
print(f"Bounding Area: {bounding_area}")
print(f"Extent: {extent:.4f}")

# Draw bounding rectangle
cv2.rectangle(
    image,
    (x, y),
    (x + w, y + h),
    (0, 255, 0),
    2
)

# Display extent value
cv2.putText(
    image,
    f"Extent: {extent:.4f}",
    (x, y - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

cv2.imshow("Contour Extent", image)
cv2.waitKey(0)
cv2.destroyAllWindows()