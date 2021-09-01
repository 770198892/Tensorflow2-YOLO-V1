import numpy as np
import tensorflow as tf
import tf_slim
from tensorflow.keras.layers import Dense, Flatten, Conv2D,MaxPool2D,Reshape
from tensorflow.keras import Model


def leaky_relu(features,alpha=0.1):
    return tf.nn.leaky_relu(features, alpha=alpha)

class YOLONetModel(Model):

    def __init__(self):
        super(YOLONetModel,self).__init__()
        self.conv1 = Conv2D(64,7,strides=2,input_shape=[448,448,3],padding='VALID',activation=leaky_relu)
        self.pool1 = MaxPool2D(2,2,padding='SAME')
        self.conv2 = Conv2D(192,3,padding='SAME',activation=leaky_relu)
        self.pool2 = MaxPool2D(2,2,padding='SAME')
        self.conv3 = Conv2D(128,1,padding='SAME',activation=leaky_relu)
        self.conv4 = Conv2D(256,3,padding='SAME',activation=leaky_relu)
        self.conv5 = Conv2D(256,1,padding='SAME',activation=leaky_relu)
        self.conv6 = Conv2D(512,3,padding='SAME',activation=leaky_relu)
        self.pool3 = MaxPool2D(2,2,padding='SAME')
        self.conv7 = Conv2D(256,1,padding='SAME',activation=leaky_relu)
        self.conv8 = Conv2D(512,3,padding='SAME',activation=leaky_relu)
        self.conv9 = Conv2D(256,1,padding='SAME',activation=leaky_relu)
        self.conv10 = Conv2D(512,3,padding='SAME',activation=leaky_relu)
        self.conv11 = Conv2D(256,1,padding='SAME',activation=leaky_relu)
        self.conv12 = Conv2D(512,3,padding='SAME',activation=leaky_relu)
        self.conv13 = Conv2D(256,1,padding='SAME',activation=leaky_relu)
        self.conv14 = Conv2D(512,3,padding='SAME',activation=leaky_relu)
        self.conv15 = Conv2D(512,1,padding='SAME',activation=leaky_relu)
        self.conv16 = Conv2D(1024,3,padding='SAME',activation=leaky_relu)
        self.pool4 = MaxPool2D(2,2,padding='SAME')
        self.conv17 = Conv2D(512,1,padding='SAME',activation=leaky_relu)
        self.conv18 = Conv2D(1024,3,padding='SAME',activation=leaky_relu)
        self.conv19 = Conv2D(512,1,padding='SAME',activation=leaky_relu)
        self.conv20 = Conv2D(1024,3,padding='SAME',activation=leaky_relu)
        self.conv21 = Conv2D(1024,3,padding='SAME',activation=leaky_relu)
        self.conv22 = Conv2D(1024,3,2,padding='VALID',activation=leaky_relu)
        self.conv23 = Conv2D(1024,3,padding='SAME',activation=leaky_relu)
        self.conv24 = Conv2D(1024,3,padding='SAME',activation=leaky_relu)
        self.flatten1 = Flatten()
        self.dense1 = Dense(512,activation=leaky_relu)
        self.dense2 = Dense(4096,activation=leaky_relu)
        self.drop = tf.keras.layers.Dropout(0.5)
        self.dense3 = tf.keras.layers.Dense(1470,activation=None)


    def call(self, inputs):
        # 数据填充
        inputs = tf.pad(inputs,np.array([[0,0],[3,3],[3,3],[0,0]]))
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.pool4(x)
        x = self.conv17(x)
        x = self.conv18(x)
        x = self.conv19(x)
        x = self.conv20(x)
        x = self.conv21(x)
        x = tf.pad(x, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]))
        x = self.conv22(x)
        x = self.conv23(x)
        x = self.conv24(x)
        x = tf.transpose(x, [0, 3, 1, 2])
        x = self.flatten1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        #x = tf_slim.dropout(x,keep_prob=0.5,is_training=False)
        x = self.drop(x)
        x = self.dense3(x)
        return x

