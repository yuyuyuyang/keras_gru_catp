import os
import tensorflow as tf
from PIL import Image  # 注意Image,后面会用到
import matplotlib.pyplot as plt
import numpy as np
import sys
import base64
import io
import cv2

# 识别字符集
char_ocr = '0123456789abcdefghijklmnopqrstuvwxyz'  # string.digits
# 定义识别字符串的最大长度
seq_len = 4
# 识别结果集合个数 0-9
label_count=len(char_ocr)+1
# 图片文件路径
# cwd = './data/'# 图片文件路径
# tfrecord_path = "./tf_recordes/dog_test.tfrecords"# tfrecords文件路径

cwd = './data_onece/'# 图片文件路径
tfrecord_path = "./tf_recordes/dog_train.tfrecords"# tfrecords文件路径

def make_da():
    writer = tf.python_io.TFRecordWriter(tfrecord_path)  # 要生成的文件
    for i,j in enumerate(os.listdir(cwd)):
        name_l = j.split('_')[0]
        print(j)

        lab = []
        for num in str(name_l):
            lab.append(int(char_ocr.find(num)))
            # print(lab)
        if len(lab) < seq_len:
            cur_seq_len = len(lab)
            for i in range(seq_len - cur_seq_len):
                lab.append(label_count)
        name_l2 = np.array(lab)
        print(name_l2)
        img_path = cwd + j
        print(img_path)

        img = cv2.imread(img_path, 0)
        # print(img)
        # cv2.namedWindow("the window")
        # cv2.imshow("the window",img)
        img = cv2.resize(img, (150, 60), interpolation=cv2.INTER_CUBIC)
        img = cv2.transpose(img,(60,150))
        img =cv2.flip(img,1)
        # cv2.namedWindow("the window")
        # cv2.imshow("the window",img)
        # cv2.waitKey()
        img = (255 - img) / 256  # 反色处理

        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[name_l2.astype(np.float64).tostring()])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.astype(np.float64).tostring()]))
        }))  # example对象对label和image数据进行封装
        writer.write(example.SerializeToString())
    writer.close()


if __name__ =='__main__':
    make_da()