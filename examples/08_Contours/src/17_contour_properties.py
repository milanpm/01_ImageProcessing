import cv2
import math

# Load image
image = cv2.imread("images/sample.png")

if image is None:
    raise FileNotFoundError("Could not load images/sample.png")

print("Image loaded successfully.")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Binary threshold
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print(f"Contours found: {len(contours)}")


# Select the largest contour
largest_contour = max(contours, key=cv2.contourArea)

area = cv2.contourArea(largest_contour)

print(f"Largest contour area: {area:.1f}")


# Bounding rectangle
x, y, w, h = cv2.boundingRect(largest_contour)

# Aspect ratio
aspect_ratio = w / h

print(f"Bounding rectangle: x={x}, y={y}, w={w}, h={h}")
print(f"Aspect ratio: {aspect_ratio:.3f}")


# Extent
rect_area = w * h
extent = area / rect_area

print(f"Rectangle area: {rect_area}")
print(f"Extent: {extent:.3f}")


# Solidity
hull = cv2.convexHull(largest_contour)
hull_area = cv2.contourArea(hull)
solidity = area / hull_area

print(f"Convex hull area: {hull_area:.1f}")
print(f"Solidity: {solidity:.3f}")


# Equivalent diameter
equivalent_diameter = math.sqrt(4 * area / math.pi)

print(f"Equivalent diameter: {equivalent_diameter:.2f}")


# Draw contour and bounding rectangle
result = image.copy()

cv2.drawContours(result, [largest_contour], -1, (0, 255, 0), 2)
cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow("Original", image)
cv2.imshow("Contour Properties", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
