"""
File: 06_top_hat.py
Author: Alex
Created: 2026-09-03
Last Updated: 2026-09-03

Description:
    Demonstrates the Top-Hat transformation using OpenCV.

    The Top-Hat transformation extracts small bright regions from an
    image by subtracting the morphological opening from the original
    grayscale image:

        Top-Hat = Original - Opening

    Morphological opening removes bright structures that are smaller
    than the structuring element. Subtracting the opened image from the
    original image reveals the removed bright details.

Processing Steps:
    1. Load the source image.
    2. Convert the image to grayscale.
    3. Create a 15 x 15 elliptical structuring element.
    4. Apply morphological opening.
    5. Calculate the Top-Hat transformation.
    6. Verify that Top-Hat equals the original minus the opening.
    7. Measure the extracted bright pixels.
    8. Save and display the result.

Input:
    images/sample.png

Output:
    outputs/06_Morphology/top_hat_15x15.png
"""

import cv2
import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
image_path = ROOT / "images" / "sample.png"
output_dir = ROOT / "outputs" / "06_Morphology"
output_path = output_dir / "top_hat_15x15.png"

# Load the source image.
image = cv2.imread(str(image_path))

if image is None:
    print(f"Error: Image file not found: {image_path}")
    raise SystemExit

# Convert the color image to grayscale.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a 15 x 15 elliptical structuring element.
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (15, 15)
)

# Apply morphological opening.
opening = cv2.morphologyEx(
    gray,
    cv2.MORPH_OPEN,
    kernel
)

# Apply the Top-Hat transformation.
top_hat = cv2.morphologyEx(
    gray,
    cv2.MORPH_TOPHAT,
    kernel
)

# Verify the definition: Top-Hat = original - opening.
expected_top_hat = cv2.subtract(gray, opening)
results_match = np.array_equal(top_hat, expected_top_hat)

# Create the output directory when it does not already exist.
output_dir.mkdir(parents=True, exist_ok=True)

# Save the Top-Hat result.
if not cv2.imwrite(str(output_path), top_hat):
    print(f"Error: Failed to save the result: {output_path}")
    raise SystemExit

# Measure the extracted bright pixels.
total_pixels = top_hat.size
extracted_pixels = cv2.countNonZero(top_hat)
extracted_percentage = extracted_pixels / total_pixels * 100
maximum_intensity = int(top_hat.max())
mean_intensity = float(top_hat.mean())

print(f"Kernel shape: Ellipse")
print(f"Kernel size: {kernel.shape[1]} x {kernel.shape[0]}")
print(f"Image size: {top_hat.shape[1]} x {top_hat.shape[0]}")
print(f"Total pixels: {total_pixels}")
print(f"Extracted bright pixels: {extracted_pixels}")
print(f"Extracted percentage: {extracted_percentage:.2f}%")
print(f"Maximum Top-Hat intensity: {maximum_intensity}")
print(f"Mean Top-Hat intensity: {mean_intensity:.2f}")
print(f"Top-Hat equals original minus opening: {results_match}")
print(f"Saved result: {output_path}")

# Display the original grayscale image, opening, and Top-Hat result.
cv2.imshow("Original Grayscale Image", gray)
cv2.imshow("Morphological Opening", opening)
cv2.imshow("Top-Hat Transformation", top_hat)

cv2.waitKey(0)
cv2.destroyAllWindows()
