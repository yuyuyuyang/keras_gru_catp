import numpy as np
from PIL import Image
import random
import re
import os
import sys
import cv2
import tensorflow as tf


# 识别字符集
char_ocr = '0123456789abcdefghijklmnopqrstuvwxyz'  # string.digits
# 定义识别字符串的最大长度
seq_len = 4
# 识别结果集合个数 0-9
label_count=len(char_ocr)+1

img_weight = 150
img_height = 60


# 获取文件labels
def get_label(filepath):
    # print(str(os.path.split(filepath)[-1]).split('.')[0].split('_')[-1])
    lab=[]
    for num in str(os.path.split(filepath)[-1]).split('.')[0].split('_')[0]:
        lab.append(int(char_ocr.find(num)))
    # print(lab)
    if len(lab) < seq_len:
        cur_seq_len = len(lab)
        for i in range(seq_len - cur_seq_len):
            lab.append(label_count)
    # print(lab)
    return lab


# 训练集处理图片
def gen_image_data(dir=r'data_onece/', file_list=[]):
    dir_path = dir
    for rt, dirs, files in os.walk(dir_path):  # =pathDir
        for filename in files:
            # print (filename)
            if filename.find('.') >= 0:
                (shotname, extension) = os.path.splitext(filename)
                # print shotname,extension
                if extension == '.jpg':  # extension == '.png' or
                    file_list.append(os.path.join('%s\\%s' % (rt, filename)))
                    # print (filename)

    print(len(file_list))
    index = 0
    X = []
    Y = []
    for file in file_list:

        index += 1
        # if index>1000:
        #     break
        # print(file)
        img = cv2.imread(file, 0)
        # print(np.shape(img))
        # cv2.namedWindow("the window")
        # cv2.imshow("the window",img)
        img = cv2.resize(img, (img_weight, img_height), interpolation=cv2.INTER_CUBIC)
        img = cv2.transpose(img,(img_height,img_weight))
        img =cv2.flip(img,1)
        # cv2.namedWindow("the window")
        # cv2.imshow("the window",img)
        # cv2.waitKey()
        img = (255 - img) / 256  # 反色处理
        X.append([img])
        Y.append(get_label(file))
        # print(get_label(file))
        # print(np.shape(X))
        # print(np.shape(X))

    # print(np.shape(X))
    X = np.transpose(X, (0, 2, 3, 1))
    X = np.array(X)
    Y = np.array(Y)
    print('XXXXXXXXX',X.shape)
    return X, Y

# 预测图片处理
def gen_image_data2(image):
    X = []
    # img = cv2.imread(file, 0)
    r,g,b = cv2.split(np.array(image))
    img = cv2.merge([b, g, r])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print('imgggg',img)
    # sys.exit()
    # print(np.shape(img))
    # cv2.namedWindow("the window")
    # cv2.imshow("the window",img)
    img = cv2.resize(img, (img_weight, img_height), interpolation=cv2.INTER_CUBIC)
    img = cv2.transpose(img, (img_height, img_weight))
    img = cv2.flip(img, 1)  # 图片翻转
    # cv2.namedWindow("the window")
    # cv2.imshow("the window",img)
    # cv2.waitKey()
    img = (255 - img) / 256  # 反色处理
    X.append([img])
    # Y.append(get_label(file))
    # print(get_label(file))
    # print('ccccc',np.shape(X))
    # print(np.shape(X))

    # print(np.shape(X))
    X = np.transpose(X, (0, 2, 3, 1))
    # X = np.array(X)

    print('XXXXXXXXX', X.shape)
    print(type(X))
    return X




def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.string),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.float64)
    img = tf.reshape(img, [150, 60, 1])
    img = tf.cast(img, tf.float32)
    label = tf.decode_raw(features['label'], tf.float64)
    label = tf.reshape(label, [4, ])
    label = tf.cast(label, tf.int8)
    # print(img, label)
    return img, label



def datass_c(tfrecord_path, batch_num):
    img, label = read_and_decode(tfrecord_path)
    # print(img.shape)
    #使用shuffle_batch可以随机打乱输入
    # img_batch, label_batch = tf.train.shuffle_batch([img, label],
    #                                                 batch_size=100, capacity=2000,
    #                                                 min_after_dequeue=1000)
    img_batch, label_batch = tf.train.batch([img, label], batch_size=batch_num, capacity=2000)
    # print(img_batch, label_batch )
    init = tf.global_variables_initializer()

    sess = tf.Session()
    # with tf.Session() as sess:
    sess.run(init)

    # 开启一个协调器
    coord = tf.train.Coordinator()
    # 使用start_queue_runners 启动队列填充
    threads = tf.train.start_queue_runners(sess, coord)

    try:
        while not coord.should_stop():
            print('************')
            # 获取每一个batch中batch_size个样本和标签
            val, l = sess.run([img_batch, label_batch])
            return val, l
    except tf.errors.OutOfRangeError:  # 如果读取到文件队列末尾会抛出此异常
        print("done! now lets kill all the threads……")
    finally:
        # 协调器coord发出所有线程终止信号
        coord.request_stop()
        print('all threads are asked to stop!')
    coord.join(threads)  # 把开启的线程加入主线程，等待threads结束
    sess.close()

    # 我们也可以根据需要对val， l进行处理
    # l = to_categorical(l, 12)
    # print(val, l)
    # print(val.shape, l.shape)
    # except tf.errors.OutOfRangeError:
    #     break















