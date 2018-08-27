#CNN输入数据整理

from itertools import groupby
import tensorflow as tf
import cv2
import numpy as np
import os
import random
import sys


from sklearn.model_selection import train_test_split
import pandas as pd

#参数设置
size=64
imgs = []
labs = []
img_path=[]
#人脸路径
input_dir= './my_faces'
def my_faces(input_dir):
    for (path, dirnames, filenames) in os.walk(input_dir):
        for dirname in dirnames:
            img_path.append(path+'/'+dirname)
        return img_path

my_faces_path= my_faces(input_dir)
other_faces_path =['./other_faces']
#padding格式
def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0,0,0,0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

def readData(paths , h=size, w=size):
    for path in paths:
        for filename in os.listdir(path):
            if filename.endswith('.jpg'):
                filename = path + '/' + filename

                img = cv2.imread(filename)

                top,bottom,left,right = getPaddingSize(img)
                # 将图片放大， 扩充图片边缘部分
                img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
                img = cv2.resize(img, (h, w))

                imgs.append(img)
                labs.append(path)
def make_one_hot(data):
    return (numpy.arange(10)==data[:,None]).astype(numpy.integer)
                
#数据录入处理
readData(my_faces_path)
readData(other_faces_path)
for lab in labs:
    for i in range(len(my_faces_path)):
        if lab ==  my_faces_path[i]:
            lab=i+1
          
        elif lab == other_faces_path[0]:
            lab = 0
            
imgs = np.array(imgs)

data_dummy=pd.get_dummies(labs)
labs = np.array(data_dummy)
# 随机划分测试集与训练集
train_x,test_x,train_y,test_y = train_test_split(imgs, labs, test_size=0.4, random_state=random.randint(0,100))
# 参数：图片数据的总数，图片的高、宽、通道
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)
# 将数据转换成小于1的数
train_x = train_x.astype('float32')/255.0
test_x = test_x.astype('float32')/255.0
print('train size:%s, test size:%s' % (len(train_x), len(test_x)))


batch_size = 25
num_batch = len(train_x) // batch_size
name=[]
for n in os.listdir(input_dir):
    name.append(n)
labels=np.array(name) 

x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, labels.shape[0]+1])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

#权重
def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)
#偏置
def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)
#卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
#
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
    Wf = weightVariable([8*8*64, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8*8*64])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weightVariable([512,labels.shape[0]+1])
    bout = weightVariable([labels.shape[0]+1])
    #out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out

def cnnTrain():
    out = cnnLayer()

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))

    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    # 比较标签是否相等，再求的所有数的平均值，tf.cast(强制转换类型)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))
    # 将loss与accuracy保存以供tensorboard使用
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    # 数据保存器的初始化
 
    predict = tf.argmax(out, 1) 
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('./tmp', graph=tf.get_default_graph())
       
        
        for n in range(100):
            for i in range(num_batch):
                batch_x = train_x[i*batch_size : (i+1)*batch_size]
                batch_y = train_y[i*batch_size : (i+1)*batch_size]
                # 开始训练数据，同时训练三个变量，返回三个数据
                _,loss,summary = sess.run([train_step, cross_entropy, merged_summary_op],
                                           feed_dict={x:batch_x,y_:batch_y, keep_prob_5:0.5,keep_prob_75:0.75})
                summary_writer.add_summary(summary, n*num_batch+i)
                # 打印损失
                print(n*num_batch+i, loss)

                if (n*num_batch+i) % 100 == 0:
                    # 获取测试数据的准确率
                    acc = accuracy.eval({x:test_x, y_:test_y, keep_prob_5:1.0, keep_prob_75:1.0})
                    print(n*num_batch+i, acc)
                    # 准确率大于0.98时保存并退出
                    if acc > 0.98 and n > 10:
                        saver.save(sess, './model/train_faces.model', global_step=n*num_batch+i)
                        print ('saver done')
                        sys.exit(0)

if __name__ == '__main__':
    cnnTrain()