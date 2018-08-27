import os
import random
import numpy as np
import cv2
from imutils.video import VideoStream
import imutils

def createdir(*args):
 
    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)


def getfacefromcamera(outdir):
    createdir(outdir)
    net = cv2.dnn.readNetFromCaffe('./deploy.prototxt', './res10_300x300_ssd_iter_140000.caffemodel')
    vs = VideoStream(src=1).start()

    n = 1
    while 1:
        if (n <= 500):
            frame = vs.read()
            frame = imutils.resize(frame, width=400)
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence < 0.5:
                    continue    
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                face = frame[startY:endY, startX:endX]
                face = cv2.resize(face, (64, 64))
                cv2.imwrite(os.path.join(outdir, str(n)+'.jpg'), face)
                n+=1    
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break 
                
                
    cv2.destroyAllWindows()
if __name__ == '__main__':
    name = input('please input yourename: ')
    getfacefromcamera(os.path.join('./my_faces', name))