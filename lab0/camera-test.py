import numpy as np
import cv2

cap = cv2.VideoCapture(0)
cv2.waitKey(60)

ret, frame = cap.read()

img = cv2.imwrite('picture.png',frame)
imgblack = cv2.imread('picture.png', 0)
height, width = imgblack.shape[:2]

imgblack = cv2.cvtColor(imgblack,cv2.COLOR_GRAY2RGB)

cv2.line(imgblack,(0,0),(width,height), (0, 255, 0), 3)
cv2.rectangle(imgblack,(width/4,height/4),(width*3/4,height*3/4),(0,255,255),3)

cv2.imwrite('black_picture.png',imgblack)

cap.release()
cv2.destroyAllWindows()
