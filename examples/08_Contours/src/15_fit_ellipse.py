import cv2
import numpy as np


image = np.zeros((500, 500, 3), dtype=np.uint8)

cv2.ellipse(
    image,
    (250, 250),
    (150, 80),
    30,
    0,
    360,
    (255, 255, 255),
    -1
)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

contours, _ = cv2.findContours(
    gray,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_NONE
)

contour = contours[0]


if len(contour) >= 5:
    fitted_ellipse = cv2.fitEllipse(contour)

    # 윤곽선에 맞춘 타원: 초록색
    cv2.ellipse(
        image,
        fitted_ellipse,
        (0, 255, 0),
        2
    )
    
    center = (
        int(fitted_ellipse[0][0]),
        int(fitted_ellipse[0][1])
    )

    # 타원의 중심점: 빨간색
cv2.circle(image, center, 5, (0, 0, 255), -1)

cv2.imshow("Fitted Ellipse", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
    
    





