"""
File: 35_center_alignment_inspection.py
Date: 2026-09-14
Author: Alex
Description:
    Inspects object alignment by comparing a contour centroid
    with a reference position. Demonstrates PASS and FAIL cases
    using a Euclidean distance tolerance in pixels.
"""

import math

import cv2
import numpy as np


REFERENCE_CENTER = (200, 220)
TOLERANCE_PX = 20.0


def inspect_alignment(label, object_center):
    """Create an object, measure its centroid, and inspect alignment."""
    image = np.zeros((400, 400, 3), dtype=np.uint8)

    # Create the same rectangular object at different positions.
    cx, cy = object_center
    cv2.rectangle(
        image,
        (cx - 60, cy - 40),
        (cx + 60, cy + 40),
        (255, 255, 255),
        -1,
    )

    # Extract the contour before drawing inspection annotations.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        raise ValueError("No contours were found.")

    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        raise ValueError("Cannot calculate centroid: contour area is zero.")

    centroid = (
        moments["m10"] / moments["m00"],
        moments["m01"] / moments["m00"],
    )

    # Positive dx means right; positive dy means down.
    dx = centroid[0] - REFERENCE_CENTER[0]
    dy = centroid[1] - REFERENCE_CENTER[1]
    distance = math.hypot(dx, dy)

    passed = distance <= TOLERANCE_PX
    status = "PASS" if passed else "FAIL"
    color = (0, 255, 0) if passed else (0, 0, 255)
    centroid_draw = tuple(map(round, centroid))

    cv2.drawContours(image, [contour], -1, color, 2)

    # The circle represents allowed centroid positions.
    cv2.circle(
        image,
        REFERENCE_CENTER,
        round(TOLERANCE_PX),
        (0, 255, 255),
        2,
    )

    cv2.line(
        image,
        REFERENCE_CENTER,
        centroid_draw,
        (255, 0, 255),
        2,
    )

    cv2.drawMarker(
        image,
        REFERENCE_CENTER,
        (255, 0, 0),
        cv2.MARKER_CROSS,
        20,
        2,
    )
    cv2.circle(image, centroid_draw, 5, (0, 165, 255), -1)

    text_lines = [
        f"{label}: {status}",
        f"dx={dx:+.2f}, dy={dy:+.2f} px",
        f"Distance: {distance:.2f} px",
        f"Tolerance: {TOLERANCE_PX:.2f} px",
    ]

    for index, text in enumerate(text_lines):
        cv2.putText(
            image,
            text,
            (15, 30 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    cv2.putText(
        image,
        "Blue cross: reference",
        (15, 345),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 180, 100),
        1,
    )
    cv2.putText(
        image,
        "Orange dot: centroid",
        (15, 375),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 165, 255),
        1,
    )

    print(f"[{label}]")
    print(f"Reference center: {REFERENCE_CENTER}")
    print(f"Measured centroid: ({centroid[0]:.2f}, {centroid[1]:.2f})")
    print(f"Offset: dx={dx:+.2f} px, dy={dy:+.2f} px")
    print(f"Distance: {distance:.2f} px")
    print(f"Tolerance: {TOLERANCE_PX:.2f} px")
    print(f"Result: {status}")
    print()

    return image


normal_image = inspect_alignment("Normal position", (210, 215))
shifted_image = inspect_alignment("Shifted position", (245, 245))

comparison = np.hstack([normal_image, shifted_image])

cv2.imshow("Center Alignment Inspection", comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()
