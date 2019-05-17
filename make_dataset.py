import os
import tensorflow as tf
from PIL import Image  # 注意Image,后面会用到
import matplotlib.pyplot as plt
import numpy as np
import sys
import base64
import io
import cv2
from english_re.image_handle import *
from english_re.model import BuildModel
import keras.backend.tensorflow_backend as KTF


#进行配置，使用60%的GPU,这是为了保护GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
session = tf.Session(config=config)
# 设置session
KTF.set_session(session)



cwd = 'data_onece/'

# 识别字符集
char_ocr = '0123456789abcdefghijklmnopqrstuvwxyz'  # string.digits
# 定义识别字符串的最大长度
seq_len = 4
# 识别结果集合个数 0-9
label_count=len(char_ocr)+1

'''读取单个数据'''
def re_da():

    for serialized_example in tf.python_io.tf_record_iterator("tf_recordes/dog_train.tfrecords"):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        image = example.features.feature['img_raw'].bytes_list.value
        label = example.features.feature['label'].bytes_list.value
        # 可以做一些预处理之类的
        print(image, label)

batch_num = len(os.listdir('data_onece/'))
X_train, Y_train = datass_c("tf_recordes/dog_train.tfrecords", batch_num)
print('2222222222',X_train.shape, Y_train.shape)

batch_size=300
epoch_size=2000
#开始建立序列模型
model = BuildModel(X_train, Y_train, batch_size, epoch_size)
# model=BuildModel(x_train,y_train,x_test,y_test,batch_size,epoch_size)
model.run_model()



