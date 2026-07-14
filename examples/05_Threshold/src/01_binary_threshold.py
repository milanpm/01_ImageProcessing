import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
image_path = ROOT / "images" / "sample.png"

image = cv2.imread(str(image_path))

if image is None:
    print("Error: Image file not found.")
    raise SystemExit

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

threshold_value = 127

_, binary = cv2.threshold(
    gray,
    threshold_value,
    255,
    cv2.THRESH_BINARY
)

cv2.imshow("Original", image)
cv2.imshow("Grayscale", gray)
cv2.imshow("Binary Threshold", binary)

cv2.waitKey(0)
cv2.destroyAllWindows()