import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
image_path = ROOT / "images" / "sample.png"

image = cv2.imread(str(image_path))

if image is None:
    print("Error: Image file not found.")
    raise SystemExit

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(
    gray,
    127,
    255,
    cv2.THRESH_BINARY
)

kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (3, 3)
)

eroded = cv2.erode(
    binary,
    kernel,
    iterations=1
)

cv2.imshow("Original", image)
cv2.imshow("Binary", binary)
cv2.imshow("Erosion", eroded)

cv2.waitKey(0)
cv2.destroyAllWindows()
