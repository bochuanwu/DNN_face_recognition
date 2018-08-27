import tensorflow as tf
import cv2
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split
from imutils.video import VideoStream
import imutils

my_faces_path='./my_faces'
size = 64
name=[]
for n in os.listdir(my_faces_path):
    name.append(n)
labels=np.array(name) 

x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, labels.shape[0]+1])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def dropout(x, keep):
    return tf.nn.dropout(x, keep)

def cnnLayer():
    # 第一层
    W1 = weightVariable([3,3,3,32]) # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = biasVariable([32])
    # 卷积
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    # 池化
    pool1 = maxPool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5)

    # 第二层
    W2 = weightVariable([3,3,32,64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    # 第三层
    W3 = weightVariable([3,3,64,64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

    # 全连接层
    Wf = weightVariable([8*16*32, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8*16*32])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weightVariable([512,labels.shape[0]+1])
    bout = biasVariable([labels.shape[0]+1])
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out

output = cnnLayer()  
predict = tf.argmax(output, 1)  
   
saver = tf.train.Saver()  
sess = tf.Session()  
saver.restore(sess, tf.train.latest_checkpoint('./model'))  
   
def is_my_face(image):  
    res = sess.run(predict, feed_dict={x: [image/255.0], keep_prob_5:1.0, keep_prob_75: 1.0})  
     
    return res[0]


net = cv2.dnn.readNetFromCaffe('./deploy.prototxt', './res10_300x300_ssd_iter_140000.caffemodel')
vs = VideoStream(src=1).start()
while 1:  
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
        y = startY - 10 if startY - 10 > 10 else startY + 10
        face = frame[startY:endY, startX:endX]
        face = cv2.resize(face, (64, 64))

        print('Is this my face? %s' % is_my_face(face))
       
        for j in range(labels.shape[0]):
            if is_my_face(face)==j:
                
                cv2.putText(frame, 'this is %s'% name[j],(startX, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        img = cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
    cv2.imshow('image',img)
                   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            

cv2.destroyAllWindows()
sess.close() 
sys.exit(0)