"""
Example 01

Load an image using OpenCV
and print basic image information.

Author : Alex
"""

import cv2

image_path = "images/sample.png"

img = cv2.imread(image_path)

if img is None:
    print("Error: image file not found.")
    exit()

height, width = img.shape[:2]

if len(img.shape) == 3:
    channels = img.shape[2]
else:
    channels = 1

print(f"Width    : {width}")
print(f"Height   : {height}")
print(f"Channels : {channels}")

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
