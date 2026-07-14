import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
image_path = ROOT / "images" / "sample.png"

image = cv2.imread(str(image_path))

if image is None:
    print("Error: Image file not found.")
    raise SystemExit

average_blur = cv2.blur(image, (5, 5))

cv2.imshow("Original", image)
cv2.imshow("Average Blur", average_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()