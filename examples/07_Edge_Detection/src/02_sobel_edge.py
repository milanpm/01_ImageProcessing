import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
image_path = ROOT / "images" / "sample.png"

image = cv2.imread(str(image_path))

if image is None:
    print("Error: Image file not found.")
    raise SystemExit

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

cv2.imshow("Original", gray)
cv2.imshow("Sobel X", sobel_x)
cv2.imshow("Sobel Y", sobel_y)
cv2.imshow("Sobel Edge", sobel)

cv2.waitKey(0)
cv2.destroyAllWindows()
