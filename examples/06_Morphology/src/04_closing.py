"""
File: 04_closing.py
Author: Alex
Created: 2026-09-01
Last Updated: 2026-09-01

Description:
    Demonstrates how to apply morphological closing to a binary image
    using OpenCV.

    Morphological closing performs two operations in sequence:

        1. Dilation
        2. Erosion

    Dilation first expands white foreground regions, filling small black
    holes and gaps. Erosion then restores the main foreground objects close
    to their original size.

Processing Steps:
    1. Load the source image.
    2. Convert the image to grayscale.
    3. Create a binary image using a threshold value of 127.
    4. Create a 5 x 5 rectangular kernel.
    5. Apply morphological closing with cv2.morphologyEx().
    6. Count the foreground pixels added by closing.
    7. Count all changed pixels.
    8. Save and display the closing result.

Input:
    images/sample.png

Output:
    outputs/06_Morphology/closing_5x5.png
"""

import cv2
import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
image_path = ROOT / "images" / "sample.png"
output_dir = ROOT / "outputs" / "06_Morphology"
output_path = output_dir / "closing_5x5.png"

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

# Apply dilation followed by erosion.
closing = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    kernel
)

# Create the output directory when it does not already exist.
output_dir.mkdir(parents=True, exist_ok=True)

# Save the processed image.
if not cv2.imwrite(str(output_path), closing):
    print(f"Error: Failed to save the result: {output_path}")
    raise SystemExit

# Count foreground pixels before and after closing.
original_white_pixels = cv2.countNonZero(binary)
closing_white_pixels = cv2.countNonZero(closing)
added_white_pixels = closing_white_pixels - original_white_pixels

# Count every pixel whose value changed.
difference = cv2.absdiff(binary, closing)
changed_pixels = cv2.countNonZero(difference)

print(f"Kernel size: {kernel.shape[1]} x {kernel.shape[0]}")
print(f"Original white pixels: {original_white_pixels}")
print(f"Closing white pixels: {closing_white_pixels}")
print(f"Added white pixels: {added_white_pixels}")
print(f"Changed pixels: {changed_pixels}")
print(f"Saved result: {output_path}")

# Display the binary input and closing result.
cv2.imshow("Binary Image", binary)
cv2.imshow("Morphological Closing", closing)

cv2.waitKey(0)
cv2.destroyAllWindows()
