import cv2
import numpy as np

image = np.zeros((500, 500, 3), dtype=np.uint8)

points = np.array([
    [100, 100],
    [400, 100],
    [400, 400],
    [300, 400],
    [250, 250],
    [200, 400],
    [100, 400]
], dtype=np.int32)

cv2.fillPoly(image, [points], (255, 255, 255))


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

contours, _ = cv2.findContours(
    gray,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

contour = contours[0]

hull = cv2.convexHull(contour, returnPoints=False)

defects = cv2.convexityDefects(contour, hull)

if defects is not None:
    for defect in defects:
        start_index, end_index, far_index, depth = defect[0]
        start = tuple(contour[start_index][0])
        end = tuple(contour[end_index][0])
        far = tuple(contour[far_index][0])
        cv2.line(image, start, end, (0, 255, 0), 2)
        cv2.circle(image, far, 6, (0, 0, 255), -1)

cv2.imshow("Convexity Defects", image)
      
cv2.waitKey(0)     
cv2.destroyAllWindows()


      
      

