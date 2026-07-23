import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
image_path = ROOT / "images" / "sample.png"

image = cv2.imread(str(image_path))

if image is None:
    print("Error: Image file not found.")
    raise SystemExit

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

result = image.copy()

cv2.drawContours(
    result,
    contours,
    -1,
    (0, 255, 0),
    2
)

print(f"Number of contours: {len(contours)}")

cv2.imshow("Original Image", image)
cv2.imshow("Binary Image", binary)
cv2.imshow("Contours", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
