"""
File: 07_black_hat.py
Author: Alex
Created: 2026-09-03
Last Updated: 2026-09-03

Description:
    Demonstrates the Black-Hat transformation using OpenCV.

    The Black-Hat transformation extracts small dark regions from an
    image by subtracting the original grayscale image from its
    morphological closing:

        Black-Hat = Closing - Original

    Morphological closing fills or suppresses dark structures that are
    smaller than the structuring element. Subtracting the original
    image from the closing reveals the affected dark details.

Processing Steps:
    1. Load the source image.
    2. Convert the image to grayscale.
    3. Create a 15 x 15 elliptical structuring element.
    4. Apply morphological closing.
    5. Calculate the Black-Hat transformation.
    6. Verify that Black-Hat equals closing minus the original.
    7. Measure the extracted dark pixels.
    8. Save and display the result.

Input:
    images/sample.png

Output:
    outputs/06_Morphology/black_hat_15x15.png
"""

import cv2
import numpy as np
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
image_path = ROOT / "images" / "sample.png"
output_dir = ROOT / "outputs" / "06_Morphology"
output_path = output_dir / "black_hat_15x15.png"

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

# Apply morphological closing.
closing = cv2.morphologyEx(
    gray,
    cv2.MORPH_CLOSE,
    kernel
)

# Apply the Black-Hat transformation.
black_hat = cv2.morphologyEx(
    gray,
    cv2.MORPH_BLACKHAT,
    kernel
)

# Verify the definition: Black-Hat = closing - original.
expected_black_hat = cv2.subtract(closing, gray)
results_match = np.array_equal(black_hat, expected_black_hat)

# Create the output directory when it does not already exist.
output_dir.mkdir(parents=True, exist_ok=True)

# Save the Black-Hat result.
if not cv2.imwrite(str(output_path), black_hat):
    print(f"Error: Failed to save the result: {output_path}")
    raise SystemExit

# Measure the extracted dark pixels.
total_pixels = black_hat.size
extracted_pixels = cv2.countNonZero(black_hat)
extracted_percentage = extracted_pixels / total_pixels * 100
maximum_intensity = int(black_hat.max())
mean_intensity = float(black_hat.mean())

print("Kernel shape: Ellipse")
print(f"Kernel size: {kernel.shape[1]} x {kernel.shape[0]}")
print(f"Image size: {black_hat.shape[1]} x {black_hat.shape[0]}")
print(f"Total pixels: {total_pixels}")
print(f"Extracted dark pixels: {extracted_pixels}")
print(f"Extracted percentage: {extracted_percentage:.2f}%")
print(f"Maximum Black-Hat intensity: {maximum_intensity}")
print(f"Mean Black-Hat intensity: {mean_intensity:.2f}")
print(f"Black-Hat equals closing minus original: {results_match}")
print(f"Saved result: {output_path}")

# Display the original grayscale image, closing, and Black-Hat result.
cv2.imshow("Original Grayscale Image", gray)
cv2.imshow("Morphological Closing", closing)
cv2.imshow("Black-Hat Transformation", black_hat)

cv2.waitKey(0)
cv2.destroyAllWindows()
