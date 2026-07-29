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
    gray, 127, 255, cv2.THRESH_BINARY
)

contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

result = image.copy()

for contour in contours:
    if cv2.contourArea(contour) < 100:
        continue

    hull = cv2.convexHull(contour)

    cv2.drawContours(
        result,
        [hull],
        -1,
        (0, 255, 0),
        2
    )

cv2.imshow("Original", image)
cv2.imshow("Convex Hull", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
