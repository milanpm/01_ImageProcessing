"""
File: 08_compare_kernel_shapes.py
Author: Alex
Created: 2026-09-04
Last Updated: 2026-09-04

Description:
    Compares Rectangle, Ellipse, and Cross structuring elements
    in an OpenCV morphological opening operation.

    All three kernels use the same 5 x 5 size and are applied to
    the same binary image. This makes it possible to observe how
    kernel shape affects noise removal, boundaries, corners, and
    thin structures.

Processing Steps:
    1. Load the source image.
    2. Convert the image to grayscale.
    3. Convert the grayscale image to a binary image.
    4. Create Rectangle, Ellipse, and Cross kernels.
    5. Apply morphological opening with each kernel.
    6. Count the remaining and removed white pixels.
    7. Create a labeled comparison image.
    8. Save and display the result.

Input:
    images/sample.png

Output:
    outputs/06_Morphology/kernel_shape_comparison_5x5.png
"""

import cv2
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
image_path = ROOT / "images" / "sample.png"
output_dir = ROOT / "outputs" / "06_Morphology"
output_path = output_dir / "kernel_shape_comparison_5x5.png"

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

kernel_size = (5, 5)

# Create three structuring elements with the same size.
rectangle_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    kernel_size
)

ellipse_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    kernel_size
)

cross_kernel = cv2.getStructuringElement(
    cv2.MORPH_CROSS,
    kernel_size
)

# Apply morphological opening with each kernel.
rectangle_result = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    rectangle_kernel
)

ellipse_result = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    ellipse_kernel
)

cross_result = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    cross_kernel
)

# Measure the number of white pixels.
original_white_pixels = cv2.countNonZero(binary)
rectangle_white_pixels = cv2.countNonZero(rectangle_result)
ellipse_white_pixels = cv2.countNonZero(ellipse_result)
cross_white_pixels = cv2.countNonZero(cross_result)

rectangle_removed = original_white_pixels - rectangle_white_pixels
ellipse_removed = original_white_pixels - ellipse_white_pixels
cross_removed = original_white_pixels - cross_white_pixels

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


binary_labeled = add_label(binary, "Original Binary")
rectangle_labeled = add_label(
    rectangle_result,
    "Rectangle 5x5"
)
ellipse_labeled = add_label(
    ellipse_result,
    "Ellipse 5x5"
)
cross_labeled = add_label(
    cross_result,
    "Cross 5x5"
)

# Arrange the four images in a 2 x 2 grid.
top_row = cv2.hconcat([
    binary_labeled,
    rectangle_labeled
])

bottom_row = cv2.hconcat([
    ellipse_labeled,
    cross_labeled
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
print("Rectangle kernel:")
print(rectangle_kernel)

print("\nEllipse kernel:")
print(ellipse_kernel)

print("\nCross kernel:")
print(cross_kernel)

# Print quantitative comparison results.
print("\nMorphological Opening Comparison")
print(f"Kernel size: {kernel_size[0]} x {kernel_size[1]}")
print(f"Image size: {binary.shape[1]} x {binary.shape[0]}")
print(f"Original white pixels: {original_white_pixels}")

print("\nRectangle:")
print(f"  Remaining white pixels: {rectangle_white_pixels}")
print(f"  Removed white pixels: {rectangle_removed}")

print("\nEllipse:")
print(f"  Remaining white pixels: {ellipse_white_pixels}")
print(f"  Removed white pixels: {ellipse_removed}")

print("\nCross:")
print(f"  Remaining white pixels: {cross_white_pixels}")
print(f"  Removed white pixels: {cross_removed}")

print("\nDifferences between results:")
print(
    "  Rectangle vs Ellipse: "
    f"{cv2.countNonZero(cv2.absdiff(rectangle_result, ellipse_result))}"
)
print(
    "  Rectangle vs Cross: "
    f"{cv2.countNonZero(cv2.absdiff(rectangle_result, cross_result))}"
)
print(
    "  Ellipse vs Cross: "
    f"{cv2.countNonZero(cv2.absdiff(ellipse_result, cross_result))}"
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
    "Comparing Morphological Kernel Shapes",
    display_image
)

cv2.waitKey(0)
cv2.destroyAllWindows()
