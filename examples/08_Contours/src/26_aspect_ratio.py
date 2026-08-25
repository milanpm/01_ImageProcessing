"""
File Name: 26_aspect_ratio.py
Created: 2026.08.25
Author: Alex
Description: Demonstrates how to calculate the aspect ratio of a contour
             using its bounding rectangle in OpenCV.
"""


import cv2
import numpy as np

# Create a blank image
image = np.zeros((500, 700, 3), dtype=np.uint8)

# Draw a white rectangle
cv2.rectangle(image, (100, 150), (400, 300), (255, 255, 255), -1)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply binary threshold
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Calculate bounding rectangle and aspect ratio
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    aspect_ratio = w / h

    print(f"Width: {w}")
    print(f"Height: {h}")
    print(f"Aspect Ratio: {aspect_ratio:.4f}")

# Draw bounding rectangle
cv2.rectangle(
    image,
    (x, y),
    (x + w, y + h),
    (0, 255, 0),
    2
)

# Display aspect ratio
cv2.putText(
    image,
    f"Aspect Ratio: {aspect_ratio:.2f}",
    (x, y - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

# Show result
cv2.imshow("Aspect Ratio", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
