"""
File: 09_compare_kernel_sizes.py
Author: Alex
Created: 2026-09-07
Last Updated: 2026-09-07

Description:
    Compares 3 x 3, 5 x 5, and 9 x 9 Ellipse structuring
    elements in an OpenCV morphological opening operation.

    All three kernels use the same Ellipse shape and are applied
    to the same binary image. This isolates the effect of kernel
    size on noise removal, boundaries, and thin structures.

Processing Steps:
    1. Load the source image.
    2. Convert the image to grayscale.
    3. Convert the grayscale image to a binary image.
    4. Create 3 x 3, 5 x 5, and 9 x 9 Ellipse kernels.
    5. Apply morphological opening with each kernel.
    6. Count the remaining and removed white pixels.
    7. Calculate removal rates and differences between results.
    8. Create a labeled comparison image.
    9. Save and display the result.

Input:
    images/sample.png

Output:
    outputs/06_Morphology/kernel_size_comparison_3x3_5x5_9x9.png
"""

import cv2
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
image_path = ROOT / "images" / "sample.png"
output_dir = ROOT / "outputs" / "06_Morphology"
output_path = (
    output_dir /
    "kernel_size_comparison_3x3_5x5_9x9.png"
)

# Load the source image.
image = cv2.imread(str(image_path))

if image is None:
    print(f"Error: Image file not found: {image_path}")
    raise SystemExit

# Convert the color image to grayscale.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert the grayscale image to a binary image.
_, binary = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)

# Create Ellipse structuring elements with different sizes.
kernel_3x3 = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)

kernel_5x5 = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)

kernel_9x9 = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (9, 9)
)

# Apply morphological opening with each kernel.
result_3x3 = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel_3x3
)

result_5x5 = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel_5x5
)

result_9x9 = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel_9x9
)

# Measure the number of white pixels.
original_white_pixels = cv2.countNonZero(binary)

white_pixels_3x3 = cv2.countNonZero(result_3x3)
white_pixels_5x5 = cv2.countNonZero(result_5x5)
white_pixels_9x9 = cv2.countNonZero(result_9x9)

removed_3x3 = original_white_pixels - white_pixels_3x3
removed_5x5 = original_white_pixels - white_pixels_5x5
removed_9x9 = original_white_pixels - white_pixels_9x9

# Calculate the percentage of removed white pixels.
removal_rate_3x3 = (
    removed_3x3 / original_white_pixels * 100
)
removal_rate_5x5 = (
    removed_5x5 / original_white_pixels * 100
)
removal_rate_9x9 = (
    removed_9x9 / original_white_pixels * 100
)

# Measure differences between the processed results.
difference_3x3_5x5 = cv2.countNonZero(
    cv2.absdiff(result_3x3, result_5x5)
)

difference_5x5_9x9 = cv2.countNonZero(
    cv2.absdiff(result_5x5, result_9x9)
)

difference_3x3_9x9 = cv2.countNonZero(
    cv2.absdiff(result_3x3, result_9x9)
)


# Create labeled images for visual comparison.
def add_label(source, label):
    labeled = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)

    cv2.rectangle(
        labeled,
        (0, 0),
        (labeled.shape[1], 45),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        labeled,
        label,
        (15, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return labeled


binary_labeled = add_label(
    binary,
    "Original Binary"
)

result_3x3_labeled = add_label(
    result_3x3,
    "Ellipse 3x3"
)

result_5x5_labeled = add_label(
    result_5x5,
    "Ellipse 5x5"
)

result_9x9_labeled = add_label(
    result_9x9,
    "Ellipse 9x9"
)

# Arrange the four images in a 2 x 2 grid.
top_row = cv2.hconcat([
    binary_labeled,
    result_3x3_labeled
])

bottom_row = cv2.hconcat([
    result_5x5_labeled,
    result_9x9_labeled
])

comparison = cv2.vconcat([
    top_row,
    bottom_row
])

# Create the output directory when it does not already exist.
output_dir.mkdir(parents=True, exist_ok=True)

# Save the comparison image.
if not cv2.imwrite(str(output_path), comparison):
    print(f"Error: Failed to save the result: {output_path}")
    raise SystemExit

# Print the actual kernel structures.
print("Ellipse 3 x 3 kernel:")
print(kernel_3x3)

print("\nEllipse 5 x 5 kernel:")
print(kernel_5x5)

print("\nEllipse 9 x 9 kernel:")
print(kernel_9x9)

# Print quantitative comparison results.
print("\nMorphological Opening Kernel Size Comparison")
print("Kernel shape: Ellipse")
print(f"Image size: {binary.shape[1]} x {binary.shape[0]}")
print(f"Original white pixels: {original_white_pixels}")

print("\nEllipse 3 x 3:")
print(f"  Remaining white pixels: {white_pixels_3x3}")
print(f"  Removed white pixels: {removed_3x3}")
print(f"  Removal rate: {removal_rate_3x3:.2f}%")

print("\nEllipse 5 x 5:")
print(f"  Remaining white pixels: {white_pixels_5x5}")
print(f"  Removed white pixels: {removed_5x5}")
print(f"  Removal rate: {removal_rate_5x5:.2f}%")

print("\nEllipse 9 x 9:")
print(f"  Remaining white pixels: {white_pixels_9x9}")
print(f"  Removed white pixels: {removed_9x9}")
print(f"  Removal rate: {removal_rate_9x9:.2f}%")

print("\nDifferences between results:")
print(f"  3 x 3 vs 5 x 5: {difference_3x3_5x5}")
print(f"  5 x 5 vs 9 x 9: {difference_5x5_9x9}")
print(f"  3 x 3 vs 9 x 9: {difference_3x3_9x9}")

print("\nRemoval strength:")
print(
    f"  9 x 9 ({removal_rate_9x9:.2f}%) > "
    f"5 x 5 ({removal_rate_5x5:.2f}%) > "
    f"3 x 3 ({removal_rate_3x3:.2f}%)"
)

print(f"\nSaved result: {output_path}")

# Display the combined comparison image.
# Resize only the display image.
# The saved comparison image keeps its original resolution.
display_scale = 0.5

display_image = cv2.resize(
    comparison,
    None,
    fx=display_scale,
    fy=display_scale,
    interpolation=cv2.INTER_AREA
)

cv2.imshow(
    "Comparing Morphological Kernel Sizes",
    display_image
)

cv2.waitKey(0)
cv2.destroyAllWindows()
