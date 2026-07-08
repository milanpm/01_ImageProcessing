import cv2

image_path = "images/sample.png"

img = cv2.imread(image_path)

if img is None:
    print("Error: image file not found.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("images/sample_gray.png", gray)

print("Grayscale image saved: images/sample_gray.png")
