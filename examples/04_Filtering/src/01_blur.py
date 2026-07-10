from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[3]
image_path = "images/sample.png"

img = cv2.imread(image_path)

if img is None:
    print("Error: image file not found.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

_, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("Original", img)
cv2.imshow("Gray", gray)
cv2.imshow("Blur", blur)
cv2.imshow("Threshold", thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
