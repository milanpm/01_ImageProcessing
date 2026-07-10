from pathlib import Path
import cv2
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
image_path = "images/sample.png"

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Error: image file not found.")
    exit()

plt.hist(img.ravel(), bins=256, range=[0, 256])
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()
