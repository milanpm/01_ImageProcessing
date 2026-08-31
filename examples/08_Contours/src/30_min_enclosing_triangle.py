import cv2
import numpy as np


# Create a black canvas
image = np.zeros((520, 1200, 3), dtype=np.uint8)

# Draw three sample shapes
cv2.rectangle(image, (70, 170), (230, 330), (255, 255, 255), -1)

irregular_points = np.array(
    [
        [470, 330],
        [440, 240],
        [500, 150],
        [610, 180],
        [660, 270],
        [600, 340],
    ],
    dtype=np.int32,
)
cv2.fillPoly(image, [irregular_points], (255, 255, 255))

cv2.ellipse(
    image,
    (960, 260),
    (105, 70),
    25,
    0,
    360,
    (255, 255, 255),
    -1,
)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a binary image
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find external contours
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

# Sort contours from left to right
contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])

shape_names = ["Rectangle", "Irregular", "Ellipse"]

for name, contour in zip(shape_names, contours):
    contour_area = cv2.contourArea(contour)

    # Find the minimum-area triangle enclosing the contour
    triangle_area, triangle = cv2.minEnclosingTriangle(contour)

    # Convert triangle coordinates to integer pixel coordinates
    triangle_points = np.int32(np.round(triangle.reshape(3, 2)))

    # Draw the original contour in green
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    # Draw the minimum enclosing triangle in red
    cv2.polylines(
        image,
        [triangle_points],
        True,
        (0, 0, 255),
        2,
    )

    # Calculate how much of the triangle is occupied by the contour
    occupancy_ratio = contour_area / triangle_area

    x, y, _, _ = cv2.boundingRect(contour)
    cv2.putText(
        image,
        name,
        (x, y - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )

    print(f"{name}")
    print(f"  Contour area: {contour_area:.2f}")
    print(f"  Minimum enclosing triangle area: {triangle_area:.2f}")
    print(f"  Occupancy ratio: {occupancy_ratio:.4f}")
    print(f"  Triangle vertices: {triangle_points.tolist()}")
    print()

cv2.imshow("Minimum Enclosing Triangle", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
