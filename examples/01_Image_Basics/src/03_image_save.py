from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[3]
image_path = "images/sample.png"

img = cv2.imread(image_path)

if img is None:
    print("Error: image file not found.")
    exit()

cv2.imwrite("images/output.png", img)
print("Saved: images/output.png")
