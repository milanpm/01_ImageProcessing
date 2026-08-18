import cv2
import numpy as np


image = np.zeros((500, 500, 3), dtype=np.uint8)

rect = ((250, 250), (220, 100), 30)
box = cv2.boxPoints(rect)
box = box.astype(np.int32)

cv2.drawContours(image, [box], 0, (255, 255, 255), -1)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

contours, _ = cv2.findContours(
    gray,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

contour = max(contours, key=cv2.contourArea)

M = cv2.moments(contour)

mu20 = M["mu20"]
mu02 = M["mu02"]
mu11 = M["mu11"]

angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
angle_deg = np.degrees(angle)

print(f"Orientation: {angle_deg:.2f} degrees")

cx = int(M["m10"] / M["m00"])
cy = int(M["m01"] / M["m00"])

length = 100

end_x = int(cx + length * np.cos(angle))
end_y = int(cy + length * np.sin(angle))

cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
cv2.line(image, (cx, cy), (end_x, end_y), (0, 255, 0), 3)


cv2.imshow("Contour Orientation", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

