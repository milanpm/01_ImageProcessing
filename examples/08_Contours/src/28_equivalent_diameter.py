import cv2
import numpy as np

# Create a black image
image = np.zeros((500, 500, 3), dtype=np.uint8)

# Draw a filled circle
cv2.circle(image, (250, 250), 100, (255, 255, 255), -1)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find contours
contours, _ = cv2.findContours(
    gray,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Get the largest contour
contour = max(contours, key=cv2.contourArea)

# Calculate contour area
area = cv2.contourArea(contour)

# Calculate equivalent diameter
equivalent_diameter = np.sqrt(4 * area / np.pi)

print(f"Contour Area: {area:.2f}")
print(f"Equivalent Diameter: {equivalent_diameter:.2f}")

# Draw contour
cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

cv2.imshow("Equivalent Diameter", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
