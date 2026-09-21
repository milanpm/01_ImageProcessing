"""
File: 37_position_rotation_alignment_inspection.py
Day: 50
Topic: Position and Rotation Alignment Inspection

Description:
    This example combines object-center inspection and rotation-angle
    inspection. An object passes only when both its position error and
    angle error are within the allowed tolerances.

Applications:
    - Component assembly inspection
    - PCB and connector alignment
    - Label orientation inspection
    - Packaging inspection
"""

import cv2
import numpy as np


REFERENCE_CENTER = (200, 220)
REFERENCE_ANGLE = 10.0

POSITION_TOLERANCE = 20.0
ANGLE_TOLERANCE = 8.0

IMAGE_WIDTH = 480
IMAGE_HEIGHT = 420


def normalize_angle(angle):
    """Normalize an angle to the range [-90, 90)."""
    while angle >= 90:
        angle -= 180
    while angle < -90:
        angle += 180

    return angle


def calculate_angle_error(measured_angle, reference_angle):
    """Calculate the smallest signed difference between two angles."""
    return normalize_angle(measured_angle - reference_angle)


def create_test_image(center, angle):
    """Create a binary image containing one rotated rectangular object."""
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)

    rotated_rect = (center, (150, 80), angle)
    box = cv2.boxPoints(rotated_rect)
    box = np.int32(box)

    cv2.fillConvexPoly(image, box, 255)

    return image


def inspect_alignment(binary_image, object_name):
    """Measure the object's center and angle, then determine PASS or FAIL."""
    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise ValueError(f"No contour found for {object_name}")

    contour = max(contours, key=cv2.contourArea)

    (center_x, center_y), (width, height), angle = cv2.minAreaRect(contour)

    # Keep the measured angle aligned with the rectangle's longer side.
    if width < height:
        angle += 90.0

    measured_angle = normalize_angle(angle)

    position_error = np.hypot(
        center_x - REFERENCE_CENTER[0],
        center_y - REFERENCE_CENTER[1],
    )

    angle_error = calculate_angle_error(
        measured_angle,
        REFERENCE_ANGLE,
    )

    position_pass = position_error <= POSITION_TOLERANCE
    angle_pass = abs(angle_error) <= ANGLE_TOLERANCE
    overall_pass = position_pass and angle_pass

    return {
        "name": object_name,
        "contour": contour,
        "center": (center_x, center_y),
        "angle": measured_angle,
        "position_error": position_error,
        "angle_error": angle_error,
        "position_pass": position_pass,
        "angle_pass": angle_pass,
        "overall_pass": overall_pass,
    }


def create_result_view(binary_image, result):
    """Create a color visualization of the inspection result."""
    display = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    result_color = (0, 255, 0) if result["overall_pass"] else (0, 0, 255)

    cv2.drawContours(
        display,
        [result["contour"]],
        -1,
        result_color,
        2,
    )

    measured_center = (
        int(round(result["center"][0])),
        int(round(result["center"][1])),
    )

    cv2.circle(display, REFERENCE_CENTER, 6, (255, 0, 0), -1)
    cv2.circle(display, measured_center, 6, result_color, -1)

    cv2.line(
        display,
        REFERENCE_CENTER,
        measured_center,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        display,
        result["name"],
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        display,
        f"Position error: {result['position_error']:.2f} px",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        display,
        f"Angle error: {result['angle_error']:+.2f} deg",
        (15, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        display,
        "PASS" if result["overall_pass"] else "FAIL",
        (15, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        result_color,
        3,
    )

    return display


normal_image = create_test_image(
    center=(210, 215),
    angle=14.0,
)

misaligned_image = create_test_image(
    center=(245, 245),
    angle=35.0,
)

normal_result = inspect_alignment(normal_image, "Normal Object")
misaligned_result = inspect_alignment(
    misaligned_image,
    "Misaligned Object",
)

print("Position and Rotation Alignment Inspection")
print("-" * 55)
print(f"Reference center       : {REFERENCE_CENTER}")
print(f"Reference angle        : {REFERENCE_ANGLE:.2f} degrees")
print(f"Position tolerance     : {POSITION_TOLERANCE:.2f} px")
print(f"Angle tolerance        : +/-{ANGLE_TOLERANCE:.2f} degrees")

for result in (normal_result, misaligned_result):
    print()
    print(result["name"])
    print(
        "Measured center       : "
        f"({result['center'][0]:.2f}, {result['center'][1]:.2f})"
    )
    print(f"Position error         : {result['position_error']:.2f} px")
    print(
        "Position result        : "
        f"{'PASS' if result['position_pass'] else 'FAIL'}"
    )
    print(f"Measured angle        : {result['angle']:.2f} degrees")
    print(f"Angle error           : {result['angle_error']:+.2f} degrees")
    print(
        "Angle result           : "
        f"{'PASS' if result['angle_pass'] else 'FAIL'}"
    )
    print(
        "Overall result         : "
        f"{'PASS' if result['overall_pass'] else 'FAIL'}"
    )

normal_view = create_result_view(normal_image, normal_result)
misaligned_view = create_result_view(
    misaligned_image,
    misaligned_result,
)

combined_view = np.hstack((normal_view, misaligned_view))

cv2.imshow(
    "Day 50 - Position and Rotation Alignment Inspection",
    combined_view,
)
cv2.waitKey(0)
cv2.destroyAllWindows()
