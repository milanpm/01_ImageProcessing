import cv2
import numpy as np


def analyze_shape(name, contour, color):
    """Calculate and display the contour area, convex hull area, and hull ratio."""
    contour_area = cv2.contourArea(contour)

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    if hull_area > 0:
        hull_ratio = contour_area / hull_area
    else:
        hull_ratio = 0.0

    print(f"{name}")
    print(f"  Contour Area : {contour_area:.2f}")
    print(f"  Hull Area    : {hull_area:.2f}")
    print(f"  Hull Ratio   : {hull_ratio:.4f}")
    print()

    # Create an individual visualization canvas.
    canvas = np.full((400, 400, 3), 255, dtype=np.uint8)

    # Draw the convex hull first.
    cv2.polylines(
        canvas,
        [hull],
        isClosed=True,
        color=(0, 0, 255),
        thickness=3,
    )

    # Draw the original contour over the convex hull.
    cv2.drawContours(
        canvas,
        [contour],
        contourIdx=-1,
        color=color,
        thickness=3,
    )

    cv2.putText(
        canvas,
        name,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"Hull Ratio: {hull_ratio:.4f}",
        (20, 370),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    return canvas


# A convex rectangle: contour area and hull area should be identical.
rectangle = np.array(
    [
        [80, 100],
        [320, 100],
        [320, 300],
        [80, 300],
    ],
    dtype=np.int32,
).reshape((-1, 1, 2))

# A clearly concave U-shaped polygon.
concave = np.array(
    [
        [60, 70],
        [340, 70],
        [340, 330],
        [250, 330],
        [250, 170],
        [150, 170],
        [150, 330],
        [60, 330],
    ],
    dtype=np.int32,
).reshape((-1, 1, 2))

# An irregular polygon containing several inward dents.
irregular = np.array(
    [
        [70, 110],
        [180, 60],
        [320, 90],
        [270, 180],
        [340, 290],
        [210, 260],
        [120, 330],
        [130, 220],
        [50, 190],
    ],
    dtype=np.int32,
).reshape((-1, 1, 2))

rectangle_image = analyze_shape(
    "Rectangle",
    rectangle,
    (255, 0, 0),
)

concave_image = analyze_shape(
    "Concave",
    concave,
    (0, 180, 0),
)

irregular_image = analyze_shape(
    "Irregular",
    irregular,
    (255, 0, 255),
)

comparison = np.hstack(
    [
        rectangle_image,
        concave_image,
        irregular_image,
    ]
)

cv2.imshow("Day 44 - Convex Hull Area and Hull Ratio", comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()
