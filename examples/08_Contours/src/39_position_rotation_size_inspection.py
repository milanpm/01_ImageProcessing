"""
File: 39_position_rotation_size_inspection.py
Day: 52
Topic: Position, Rotation, and Size Inspection

Description:
    This example combines object-center inspection, rotation-angle
    inspection, and object-size inspection. An object passes only when
    its position, angle, and size are all within the allowed tolerances.

Applications:
    - Component assembly inspection
    - PCB and connector alignment
    - Part size verification
    - Packaging inspection
"""

from turtle import width

import cv2
import numpy as np


REFERENCE_CENTER = (200, 220)
REFERENCE_ANGLE = 10.0
REFERENCE_LONG_SIDE = 150.0
REFERENCE_SHORT_SIDE = 80.0

POSITION_TOLERANCE = 20.0
ANGLE_TOLERANCE = 8.0
SIZE_TOLERANCE = 8.0

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


def create_test_image(center, angle, size=(150, 80)):
    """Create a binary image containing one rotated rectangular object."""
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)

    rotated_rect = (center, size, angle)
    box = cv2.boxPoints(rotated_rect)
    box = np.int32(box)

    cv2.fillConvexPoly(image, box, 255)

    return image


def inspect_object(binary_image, object_name):
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
    
    long_side = max(width, height)
    short_side = min(width, height)

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
    
    long_error = long_side - REFERENCE_LONG_SIDE
    short_error = short_side - REFERENCE_SHORT_SIDE

    position_pass = position_error <= POSITION_TOLERANCE
    angle_pass = abs(angle_error) <= ANGLE_TOLERANCE

    long_pass = abs(long_error) <= SIZE_TOLERANCE
    short_pass = abs(short_error) <= SIZE_TOLERANCE
    size_pass = long_pass and short_pass

    overall_pass = position_pass and angle_pass and size_pass

    return {
        "name": object_name,
        "contour": contour,
        "center": (center_x, center_y),
        "angle": measured_angle,
        "long_side": long_side,
        "short_side": short_side,
        "long_error": long_error,
        "short_error": short_error,
        "position_error": position_error,
        "angle_error": angle_error,
        "position_pass": position_pass,
        "angle_pass": angle_pass,
        "size_pass": size_pass,
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

wrong_size_image = create_test_image(
    center=(210, 215),
    angle=14.0,
    size=(180, 80),
)

normal_result = inspect_object(normal_image, "Normal Object")
misaligned_result = inspect_object(
    misaligned_image,
    "Misaligned Object",
)

wrong_size_result = inspect_object(
    wrong_size_image,
    "Wrong Size Object",
)

print("Position, Rotation, and Size Inspection")
print("-" * 55)
print(f"Reference center       : {REFERENCE_CENTER}")
print(f"Reference angle        : {REFERENCE_ANGLE:.2f} degrees")
print(
    f"Reference size         : "
    f"{REFERENCE_LONG_SIDE:.2f} x {REFERENCE_SHORT_SIDE:.2f} px"
)
print(f"Position tolerance     : {POSITION_TOLERANCE:.2f} px")
print(f"Angle tolerance        : +/-{ANGLE_TOLERANCE:.2f} degrees")
print(f"Size tolerance         : +/-{SIZE_TOLERANCE:.2f} px")

for result in (
    normal_result,
    misaligned_result,
    wrong_size_result,
):
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
        "Measured size          : "
        f"{result['long_side']:.2f} x {result['short_side']:.2f} px"
    )
    print(
        "Size error             : "
        f"{result['long_error']:+.2f} / "
        f"{result['short_error']:+.2f} px"
    )
    print(
        "Size result            : "
        f"{'PASS' if result['size_pass'] else 'FAIL'}"
    )
    print(
        "Overall result         : "
        f"{'PASS' if result['overall_pass'] else 'FAIL'}"
    )

normal_view = create_result_view(normal_image, normal_result)
normal_view = create_result_view(normal_image, normal_result)

misaligned_view = create_result_view(
    misaligned_image,
    misaligned_result,
)

wrong_size_view = create_result_view(
    wrong_size_image,
    wrong_size_result,
)

combined_view = np.hstack(
    (normal_view, misaligned_view, wrong_size_view)
)

cv2.imshow(
    "Day 52 - Position, Rotation, and Size Inspection",
    combined_view,
)
cv2.waitKey(0)
cv2.destroyAllWindows()
