"""
File: 36_rotation_alignment_inspection.py
Description: Inspects object rotation using contour orientation.
Author: Alex
Date: 2026-09-18
"""

import cv2
import numpy as np


IMAGE_WIDTH = 720
IMAGE_HEIGHT = 500

REFERENCE_ANGLE = 10.0
ANGLE_TOLERANCE = 8.0


def create_rotated_rectangle(center, size, angle):
    """Create contour points for a rotated rectangle."""
    rectangle = (center, size, angle)
    box = cv2.boxPoints(rectangle)
    return box.astype(np.int32)


def calculate_orientation(contour):
    """Calculate contour orientation in degrees using image moments."""
    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        raise ValueError("Contour area is zero.")

    mu20 = moments["mu20"]
    mu02 = moments["mu02"]
    mu11 = moments["mu11"]

    angle_radians = 0.5 * np.arctan2(
        2.0 * mu11,
        mu20 - mu02
    )

    angle_degrees = np.degrees(angle_radians)
    return angle_degrees


def calculate_angle_error(measured_angle, reference_angle):
    """Calculate the smallest angle error for an axis orientation."""
    error = measured_angle - reference_angle
    return (error + 90.0) % 180.0 - 90.0


def draw_orientation_line(image, center, angle, color):
    """Draw a line showing the measured contour orientation."""
    length = 75
    angle_radians = np.radians(angle)

    dx = int(length * np.cos(angle_radians))
    dy = int(length * np.sin(angle_radians))

    start_point = (center[0] - dx, center[1] - dy)
    end_point = (center[0] + dx, center[1] + dy)

    cv2.line(image, start_point, end_point, color, 3)
    cv2.circle(image, center, 5, (0, 0, 255), -1)


def inspect_rotation(image, contour, name):
    """Measure rotation and display the inspection result."""
    measured_angle = calculate_orientation(contour)
    angle_error = calculate_angle_error(
        measured_angle,
        REFERENCE_ANGLE
    )

    passed = abs(angle_error) <= ANGLE_TOLERANCE
    result = "PASS" if passed else "FAIL"
    result_color = (0, 200, 0) if passed else (0, 0, 255)

    moments = cv2.moments(contour)
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    center = (center_x, center_y)

    cv2.drawContours(image, [contour], -1, (255, 255, 255), 2)
    draw_orientation_line(
        image,
        center,
        measured_angle,
        result_color
    )

    cv2.putText(
        image,
        name,
        (center_x - 85, center_y - 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2
    )

    cv2.putText(
        image,
        f"Angle: {measured_angle:.2f} deg",
        (center_x - 85, center_y + 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        image,
        f"Error: {angle_error:+.2f} deg",
        (center_x - 85, center_y + 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        image,
        result,
        (center_x - 40, center_y + 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        result_color,
        2
    )

    print(
        f"{name}: measured={measured_angle:.2f} deg, "
        f"reference={REFERENCE_ANGLE:.2f} deg, "
        f"error={angle_error:+.2f} deg, "
        f"result={result}"
    )


image = np.zeros(
    (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
    dtype=np.uint8
)

normal_contour = create_rotated_rectangle(
    center=(180, 220),
    size=(180, 80),
    angle=14
)

rotated_contour = create_rotated_rectangle(
    center=(530, 220),
    size=(180, 80),
    angle=35
)

inspect_rotation(image, normal_contour, "Normal Object")
inspect_rotation(image, rotated_contour, "Rotated Object")

cv2.putText(
    image,
    f"Reference: {REFERENCE_ANGLE:.1f} deg",
    (20, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)

cv2.putText(
    image,
    f"Tolerance: +/- {ANGLE_TOLERANCE:.1f} deg",
    (20, 65),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)

cv2.imshow("Rotation Alignment Inspection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
