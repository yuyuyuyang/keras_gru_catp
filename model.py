import numpy as np
import pandas as pd
import keras
from keras.models import *
from keras.layers import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Input
from keras.layers import Conv2D, MaxPooling2D,merge,concatenate
from keras.layers import BatchNormalization,Activation
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras import regularizers
from keras.models import Model
import tensorflow as tf
from PIL import Image
from english_re.image_handle import *
# from keras.callbacks import ModelCheckpoint
from keras.callbacks import *
import os
import cv2


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    # y_pred = y_pred[:, 2:, :] 测试感觉没影响
    y_pred = y_pred[:, :, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

class Ctc_Decode:
    # 用tf定义一个专门ctc解码的图和会话，就不会一直增加节点了，速度快了很多
    def __init__(self ,batch_size, timestep, nclass):
        self.batch_size = batch_size
        self.timestep = timestep
        self.nclass = nclass
        self.graph_ctc = tf.Graph()
        with self.graph_ctc.as_default():
            self.y_pred_tensor = tf.placeholder(shape=[self.batch_size, self.timestep, self.nclass], dtype=tf.float32, name="y_pred_tensor")
            self._y_pred_tensor = tf.transpose(self.y_pred_tensor, perm=[1, 0, 2])  #  要把timestep 放在第一维
            self.input_length_tensor = tf.placeholder(shape=[self.batch_size, 1], dtype=tf.int32, name="input_length_tensor")
            self._input_length_tensor = tf.squeeze(self.input_length_tensor, axis=1) #  传进来的是 [batch_size,1] 所以要去掉一维
            self.ctc_decode, _ = tf.nn.ctc_greedy_decoder(self._y_pred_tensor, self._input_length_tensor)
            self.decoded_sequences = tf.sparse_tensor_to_dense(self.ctc_decode[0])
            self.ctc_sess = tf.Session(graph=self.graph_ctc)

    def ctc_decode_tf(self, args):
        y_pred, input_length = args
        decoded_sequences = self.ctc_sess.run(self.decoded_sequences,
                                     feed_dict={self.y_pred_tensor: y_pred, self.input_length_tensor: input_length})
        return decoded_sequences


class BuildModel():
    def __init__(self, x_train, y_train,batch_size, epoch_size):
        self.x_train = x_train
        self.y_train = y_train
        # self.x_test=x_test
        # self.y_test=y_test
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.maxin = int(len(x_train)*0.95)

    def run_model(self):
        # CNC3 输入
        inputs = Input((150, 60, 1))
        x = inputs
        for i in range(3):
            x = Convolution2D(32 * 2 ** i, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.01))(x)
            # x = Convolution2D(32*2**i, (3, 3), activation='relu')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Convolution2D(64, (3, 3), activation='relu', padding='same',  kernel_regularizer=regularizers.l2(0.01))(x)
        # x = Convolution2D(32*2**i, (3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        conv_shape = x.get_shape()
        # print(conv_shape)
        x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)

        x = Dense(32, activation='relu')(x)

        '''加入gru'''

        gru_1 = GRU(64, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
        gru_1b = GRU(64, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(x)
        gru1_merged = add([gru_1, gru_1b])

        gru_2 = GRU(64, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
        gru_2b = GRU(64, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
            gru1_merged)
        x = concatenate([gru_2, gru_2b])
        x = Dropout(0.5)(x)
        predictions = Dense(37, kernel_initializer='he_normal', activation='softmax')(x)
        base_model = Model(inputs=inputs, outputs=predictions)
        # base_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta', metrics=['accuracy'])

        '''CTC_loss'''
        labels = Input(name='the_labels', shape=[4], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([predictions, labels, input_length, label_length])
        model = Model(inputs=[inputs, labels, input_length, label_length], outputs=[loss_out])
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
        model.summary()

        tfrecord_path = "tf_recordes/dog_test.tfrecords"
        num_mg1 = len(os.listdir('./data/'))
        X_test, Y_test = datass_c(tfrecord_path, num_mg1)
        '''测试集测试正确率'''
        def test(base_model, ):
            file_list = []

            # X, Y = gen_image_data(data_dir, file_list)


            y_pred = base_model.predict(X_test)
            # print(X)
            # print(Y[:, :])
            # print('++++y_pred==============', y_pred.shape)
            shape = y_pred[:, :, :].shape  # 2:
            # print('++++y_predshape==============', shape)
            in_len = np.ones(shape[0]) * shape[1]
            # print('in_lenin_lenin_lenin_lenin_len',in_len)
            # in_len = np.zeros(1, dtype=np.int32)
            # in_len[0] = 162
            # r = K.ctc_decode(y_pred[:, :, :], in_len, greedy=True, beam_width=1, top_paths=1)
            # out1 = K.get_value(r[0][0])
            gg = np.zeros((len(Y_test), 1))+9
            ctc_class = Ctc_Decode(batch_size=len(Y_test), timestep=9, nclass=37)
            out1 = ctc_class.ctc_decode_tf([y_pred[:, :, :], gg])  # ctc解码
            # ctc_class = Ctc_Decode(batch_size=batch_size, timestep=img_w // 8, nclass=nclass)
            # predict_y = ctc_class.ctc_decode_tf([predict_y, batch_x['input_length']])  # ctc解码
            out = out1[:, :seq_len]  # 2:
            # for k in out:
            #     for j in k:
            #         print(char_ocr[j])
            # print()
            # print('++++++++', out)
            error_count = 0
            for i in range(len(X_test)):
                # print(file_list[i])
                # str_src = str(os.path.split(file_list[i])[-1]).split('.')[0].split('_')[0]
                str_src = ''.join([char_ocr[x] for x in Y_test[i] if x != -1])
                # print(out[i])
                str_out = ''.join([char_ocr[x] for x in out[i] if x != -1])
                # print(str_src, str_out)
                # print('-----', str_out)
                if str_src != str_out:
                    error_count += 1
                    # print('################################', error_count)
                # img = cv2.imread(file_list[i])
                # cv2.imshow('image', img)
                # cv2.waitKey()
            # print('=====正确率=====', ((num_mg1 - error_count) / num_mg1))
            return format((num_mg1 - error_count) / num_mg1, '.3f')

        '''重构losshistory'''
        class LossHistory(Callback):
            def on_train_begin(self, logs={}):
                self.losses = []

            def on_epoch_end(self, epoch, logs=None):
                model.save('./model/model_train/model_google_layer.h5')
                # base_model.save('./model/model_test/______model_google_layer.h5')
                sco_pre = test(base_model)
                print('=====test正确率=====',sco_pre)
                # sco_train = test(base_model, data_dir='./data_onece/')
                # print('=====train正确率=====',sco_train)
                if float(sco_pre)>0.5 :
                    base_model.save('./model/model_test/base_model_google_layer_' + str(sco_pre)+ '__'+ '.h5')
                # else:
                #     print('----------')

            def on_batch_end(self, batch, logs={}):
                self.losses.append(logs.get('loss'))

        filepath = './model/model_train/'+"model_epoch{epoch:03d}-val_loss{loss:.3f}.h5"
        # 有一次提升, 则覆盖一次.
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
        # checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
        history = LossHistory()
        model.fit([self.x_train[:self.maxin], self.y_train[:self.maxin], np.array(np.ones(len(self.x_train[:self.maxin]))*int(conv_shape[1])), np.array(np.ones(len(self.x_train[:self.maxin]))*seq_len)], self.y_train[:self.maxin],
                  batch_size=self.batch_size,
                  epochs=self.epoch_size,
                  validation_data =([self.x_train[self.maxin:], self.y_train[self.maxin:], np.array(np.ones(len(self.x_train[self.maxin:]))*int(conv_shape[1])), np.array(np.ones(len(self.x_train[self.maxin:]))*seq_len)], self.y_train[self.maxin:]),
                  callbacks=[history, checkpoint]
                  )

        model.save('./model/model_train/las__model_google_layer.h5')
        #通过测试集对模型进行评价。
        # score = model.evaluate([self.x_test, self.y_test, np.array(np.ones(len(self.x_test)) * int(conv_shape[1])), np.array(np.ones(len(self.x_test)) * seq_len)], self.y_test,
        #                        batch_size=self.batch_size)
        # print(score)

