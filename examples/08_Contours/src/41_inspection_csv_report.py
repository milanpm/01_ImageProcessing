"""
===============================================================================
File: 41_inspection_csv_report.py
Project: 01_ImageProcessing
Day: 54
Topic: Inspection CSV Report
Description:
    Inspect object position, rotation, and size, then save measurements,
    inspection decisions, and failure reasons to a CSV report.
Learning Objectives:
    - Convert inspection result dictionaries into CSV rows.
    - Save reports using csv.DictWriter.
    - Manage output paths using pathlib.Path.
Author: Alex
Created Date: 2026-10-19
Last Modified: 2026-10-19
===============================================================================
"""

import csv
from pathlib import Path

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

    failure_reasons = []

    if not position_pass:
        failure_reasons.append("POSITION")

    if not angle_pass:
        failure_reasons.append("ANGLE")

    if not size_pass:
        failure_reasons.append("SIZE")

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
        "failure_reasons": failure_reasons,
    }

def save_inspection_csv(results, output_path):
    """Save inspection measurements and decisions to a CSV report."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "name",
        "center_x",
        "center_y",
        "angle_deg",
        "long_side_px",
        "short_side_px",
        "position_error_px",
        "angle_error_deg",
        "long_error_px",
        "short_error_px",
        "position_result",
        "angle_result",
        "size_result",
        "overall_result",
        "failure_reasons",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            writer.writerow({
                "name": result["name"],
                "center_x": f"{result['center'][0]:.2f}",
                "center_y": f"{result['center'][1]:.2f}",
                "angle_deg": f"{result['angle']:.2f}",
                "long_side_px": f"{result['long_side']:.2f}",
                "short_side_px": f"{result['short_side']:.2f}",
                "position_error_px": f"{result['position_error']:.2f}",
                "angle_error_deg": f"{result['angle_error']:.2f}",
                "long_error_px": f"{result['long_error']:.2f}",
                "short_error_px": f"{result['short_error']:.2f}",
                "position_result": (
                    "PASS" if result["position_pass"] else "FAIL"
                ),
                "angle_result": (
                    "PASS" if result["angle_pass"] else "FAIL"
                ),
                "size_result": (
                    "PASS" if result["size_pass"] else "FAIL"
                ),
                "overall_result": (
                    "PASS" if result["overall_pass"] else "FAIL"
                ),
                "failure_reasons": (
                    ", ".join(result["failure_reasons"]) or "NONE"
                ),
            })

    return output_path



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

    failure_text = ", ".join(result["failure_reasons"]) or "NONE"

    cv2.putText(
        display,
        f"Reasons: {failure_text}",
        (15, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        result_color,
        2,
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

results = (
    normal_result,
    misaligned_result,
    wrong_size_result,
)

report_path = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "day54_inspection_report.csv"
)

saved_path = save_inspection_csv(results, report_path)
print(f"CSV report saved: {saved_path}")

print("Inspection CSV Report")
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

for result in results:
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

    failure_text = ", ".join(result["failure_reasons"]) or "NONE"
    print(f"Failure reasons        : {failure_text}")

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
    "Day 54 - Inspection CSV Report",
    combined_view,
)
cv2.waitKey(0)
cv2.destroyAllWindows()
