import numpy as np
import cv2
import glob
import time

monkeyImage = cv2.imread("monkey.JPG")
monkeyImage = cv2.cvtColor(monkeyImage, cv2.COLOR_BGR2GRAY)
files = glob.glob ("good/*.JPG") #good photos
# files = glob.glob ("bad/*.JPG") #bad photos
result = open("result.txt", "w+")

detector = cv2.AKAZE_create()
(kpsMonkey, descsMonkey) = detector.detectAndCompute(monkeyImage, None)
result.write("MONKEY_IMAGE:\nkey points: {}, descriptors: {}\n\n".format(len(kpsMonkey), descsMonkey.shape));

for myFile in files:
    processTime = time.time()
    print("Processing: " + myFile)
    image = cv2.imread(myFile)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (kps, descs) = detector.detectAndCompute(image, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descsMonkey,descs, 2)

    goodMatch = []
    for m,n in matches:
        if m.distance < 0.8*n.distance:
            goodMatch.append([m])

    matchImage = cv2.drawMatchesKnn(monkeyImage, kpsMonkey, image, kps, goodMatch[1:20], None, 2)

    cv2.imwrite(myFile + "_result"+ ".JPG", matchImage)
    result.write(myFile+":\nkey points: {}\ndescriptors: {}\nmatches: {}\ntime: {}\n\n".format(len(kps), descs.shape, len(goodMatch), time.time() - processTime))

result.close()
