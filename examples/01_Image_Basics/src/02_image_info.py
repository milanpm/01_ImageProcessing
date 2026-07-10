from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[3]
image_path = "images/sample.png"

img = cv2.imread(image_path)

if img is None:
    print("Error: image file not found.")
    exit()

print("Shape:", img.shape)
print("Height:", img.shape[0])
print("Width:", img.shape[1])
print("Channels:", img.shape[2])
print("Data type:", img.dtype)
