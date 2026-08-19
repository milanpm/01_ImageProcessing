import cv2
import numpy as np


# Create a black canvas
image = np.zeros((500, 500, 3), dtype=np.uint8)

# Draw a filled ellipse rotated by 30 degrees
center = (250, 250)
axes = (150, 75)
angle = 30

cv2.ellipse(
    image,
    center,
    axes,
    angle,
    0,
    360,
    (255, 255, 255),
    -1,
)


# Convert the image to a binary image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Detect the outer contour
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

contour = max(contours, key=cv2.contourArea)


# Calculate contour moments
moments = cv2.moments(contour)

mu20 = moments["mu20"] / moments["m00"]
mu02 = moments["mu02"] / moments["m00"]
mu11 = moments["mu11"] / moments["m00"]

covariance = np.array(
    [
        [mu20, mu11],
        [mu11, mu02],
    ],
    dtype=np.float64,
)


# Calculate the principal variances
eigenvalues = np.linalg.eigvalsh(covariance)
minor_value, major_value = eigenvalues

# Calculate eccentricity
eccentricity = np.sqrt(1.0 - minor_value / major_value)

print(f"Minor eigenvalue: {minor_value:.2f}")
print(f"Major eigenvalue: {major_value:.2f}")
print(f"Eccentricity: {eccentricity:.4f}")


# Visualize the contour and result
result = image.copy()
cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)

cv2.putText(
    result,
    f"Eccentricity: {eccentricity:.4f}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 255),
    2,
    cv2.LINE_AA,
)

cv2.imshow("Contour Eccentricity", result)
cv2.waitKey(0)
cv2.destroyAllWindows()








