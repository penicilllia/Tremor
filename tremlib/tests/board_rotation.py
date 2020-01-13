import cv2
import imutils
import numpy as np

image = cv2.imread(r'D:\Projects\Tremor\Markers\new_board.bmp')
print(image)
for angle in np.arange(0, 360, 15):
	rotated = imutils.rotate_bound(image, angle)
	cv2.imwrite(r"D:\Projects\Tremor\Markers\Board roation\Angle %03.2f.bmp" % angle, rotated)