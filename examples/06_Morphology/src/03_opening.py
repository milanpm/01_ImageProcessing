import cv2
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
image_path = ROOT / "images" / "sample.png"

image = cv2.imread(str(image_path))

if image is None:
    print("Error: Image file not found.")
    raise SystemExit

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel = np.ones((5, 5), np.uint8)

opening = cv2.morphologyEx(
    gray,
    cv2.MORPH_OPEN,
    kernel
)

cv2.imshow("Original", gray)
cv2.imshow("Opening", opening)

cv2.waitKey(0)
cv2.destroyAllWindows()
