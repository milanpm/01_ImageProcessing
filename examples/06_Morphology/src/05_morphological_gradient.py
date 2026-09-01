"""
File: 05_morphological_gradient.py
Author: Alex
Created: 2026-09-01
Last Updated: 2026-09-01

Description:
    Demonstrates how to calculate the morphological gradient of a binary
    image using OpenCV.

    The morphological gradient is the difference between dilation and
    erosion:

        Morphological Gradient = Dilation - Erosion

    Dilation expands the white foreground, while erosion shrinks it.
    Subtracting the eroded image from the dilated image highlights the
    boundaries of foreground objects.

Processing Steps:
    1. Load the source image.
    2. Convert the image to grayscale.
    3. Create a binary image using a threshold value of 127.
    4. Create a 5 x 5 rectangular kernel.
    5. Apply dilation and erosion for comparison.
    6. Calculate the gradient with cv2.MORPH_GRADIENT.
    7. Verify that the result equals dilation minus erosion.
    8. Count the boundary pixels.
    9. Save and display the gradient result.

Input:
    images/sample.png

Output:
    outputs/06_Morphology/morphological_gradient_5x5.png
"""

import cv2
import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
image_path = ROOT / "images" / "sample.png"
output_dir = ROOT / "outputs" / "06_Morphology"
output_path = output_dir / "morphological_gradient_5x5.png"

# Load the source image.
image = cv2.imread(str(image_path))

if image is None:
    print(f"Error: Image file not found: {image_path}")
    raise SystemExit

# Convert the color image to grayscale.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert the grayscale image to a binary image.
# Pixels greater than 127 become white, and the others become black.
_, binary = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)

# Create a 5 x 5 rectangular structuring element.
kernel = np.ones((5, 5), dtype=np.uint8)

# Calculate dilation and erosion separately for verification.
dilated = cv2.dilate(
    binary,
    kernel,
    iterations=1
)

eroded = cv2.erode(
    binary,
    kernel,
    iterations=1
)

# Calculate the morphological gradient.
gradient = cv2.morphologyEx(
    binary,
    cv2.MORPH_GRADIENT,
    kernel
)

# Verify the definition: gradient = dilation - erosion.
expected_gradient = cv2.subtract(dilated, eroded)
results_match = np.array_equal(gradient, expected_gradient)

# Create the output directory when it does not already exist.
output_dir.mkdir(parents=True, exist_ok=True)

# Save the gradient image.
if not cv2.imwrite(str(output_path), gradient):
    print(f"Error: Failed to save the result: {output_path}")
    raise SystemExit

# Measure the number of highlighted boundary pixels.
total_pixels = gradient.size
boundary_pixels = cv2.countNonZero(gradient)
boundary_percentage = boundary_pixels / total_pixels * 100

print(f"Kernel size: {kernel.shape[1]} x {kernel.shape[0]}")
print(f"Image size: {gradient.shape[1]} x {gradient.shape[0]}")
print(f"Total pixels: {total_pixels}")
print(f"Boundary pixels: {boundary_pixels}")
print(f"Boundary percentage: {boundary_percentage:.2f}%")
print(f"Gradient equals dilation minus erosion: {results_match}")
print(f"Saved result: {output_path}")

# Display the binary input and morphological gradient.
cv2.imshow("Binary Image", binary)
cv2.imshow("Morphological Gradient", gradient)

cv2.waitKey(0)
cv2.destroyAllWindows()
