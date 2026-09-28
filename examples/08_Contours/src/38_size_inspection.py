"""
File: 38_size_inspection.py
Date: 2026-09-28
Author: Alex
Description:
    Measures the long and short sides of a rotated rectangular object
    and compares them with reference dimensions and a size tolerance.
"""

import cv2
import numpy as np


REFERENCE_LONG_SIDE = 150.0
REFERENCE_SHORT_SIDE = 80.0
SIZE_TOLERANCE = 8.0

IMAGE_WIDTH = 480
IMAGE_HEIGHT = 420
OBJECT_CENTER = (240, 210)


def create_test_image(size, angle=20.0):
    """Draw one rotated rectangle on a binary image."""
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
    rotated_rect = (OBJECT_CENTER, size, angle)
    box = np.int32(cv2.boxPoints(rotated_rect))
    cv2.fillConvexPoly(image, box, 255)
    return image


def inspect_size(binary_image, object_name):
    """Measure both sides and check them against the reference size."""
    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        raise ValueError(f"No contour found for {object_name}")

    contour = max(contours, key=cv2.contourArea)
    (_, _), (width, height), _ = cv2.minAreaRect(contour)

    long_side = max(width, height)
    short_side = min(width, height)
    long_error = long_side - REFERENCE_LONG_SIDE
    short_error = short_side - REFERENCE_SHORT_SIDE

    long_pass = abs(long_error) <= SIZE_TOLERANCE
    short_pass = abs(short_error) <= SIZE_TOLERANCE

    return {
        "name": object_name,
        "long_side": long_side,
        "short_side": short_side,
        "long_error": long_error,
        "short_error": short_error,
        "long_pass": long_pass,
        "short_pass": short_pass,
        "overall_pass": long_pass and short_pass,
    }


def main():
    cases = [
        ("Normal", (150, 80)),
        ("Long side too large", (175, 80)),
        ("Short side too small", (150, 60)),
    ]

    print(
        f"Reference: {REFERENCE_LONG_SIDE:.0f} x "
        f"{REFERENCE_SHORT_SIDE:.0f} px"
    )
    print(f"Tolerance: ±{SIZE_TOLERANCE:.0f} px per side\n")

    for name, size in cases:
        image = create_test_image(size)
        result = inspect_size(image, name)
        status = "PASS" if result["overall_pass"] else "FAIL"

        print(f"{name}: {status}")
        print(
            f"  Long side:  {result['long_side']:.2f} px "
            f"(error {result['long_error']:+.2f}) — "
            f"{'PASS' if result['long_pass'] else 'FAIL'}"
        )
        print(
            f"  Short side: {result['short_side']:.2f} px "
            f"(error {result['short_error']:+.2f}) — "
            f"{'PASS' if result['short_pass'] else 'FAIL'}"
        )


if __name__ == "__main__":
    main()
