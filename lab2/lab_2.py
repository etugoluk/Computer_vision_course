import numpy as np
import cv2 as cv
import time

feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0,255,(100,3))

separator = "separator"

def encode(separator):
	file = "snow_day.MOV"
	cap = cv.VideoCapture(file)

	enc_file = file[:len(file) - 4] + ".enc"
	f = open(enc_file, "wb")

	ret, old_frame = cap.read()
	old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
	p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
	mask = np.zeros_like(old_frame)

	f.write(old_frame.tobytes())
	f.write(separator.encode())

	while (1):
		ret,frame = cap.read()
		if not ret:
			break

		frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

		p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

		good_old = p0[st==1]
		good_new = p1[st==1]

		for i,(new,old) in enumerate(zip(good_new,good_old)):
			a,b = new.ravel()
			c,d = old.ravel()
			mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 2)
			frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)

		frame = cv.add(frame,mask)
		f.write(frame.tobytes())
		f.write(separator.encode())

		old_gray = frame_gray.copy()
		p0 = good_new.reshape(-1,1,2)

	cap.release()
	f.close()

	print("Video successfully encoded")
	return enc_file

def decode(encfile):

	f = open(encfile, "rb")
	data = f.read()
	f.close()

	list = data.split(separator.encode())

	for item in list[0:-1]:
		frame = np.frombuffer(item,dtype=np.float32).reshape((1080, 1440, 1))
		rgb = cv.cvtColor(frame, cv.COLOR_GRAY2BGR) #it doesn't work :(

		cv.imshow('Decoded Video', rgb)
		k = cv.waitKey(30) & 0xff
		if k == 27:
			break

	cv.destroyAllWindows()
	print("Video successfully decoded")

encfile = encode(separator)
decode(encfile)

