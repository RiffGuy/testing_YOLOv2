from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import layers, models, losses
import numpy as np

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)
#
class YOLOv2():
    def __init__(self, input_hw, anchors, classes=91):
        self.input_hw = input_hw
        self.grid_h_cnt, self.grid_w_cnt = self.input_hw[0] // 32, self.input_hw[1] // 32
        self.grid_h, self.grid_w = self.input_hw[0] // self.grid_h_cnt, self.input_hw[1] // self.grid_w_cnt
        self.anchors = anchors
        self.anchors_size = len(anchors)
        self.num_classes = classes
        self.model = self._build()

    def _conv(self, layer, filters, kernel, name):
        layer = layers.Conv2D(filters, kernel, padding='same', name="conv_"+name)(layer)
        layer = layers.BatchNormalization(name="bn_"+name)(layer)
        layer = layers.LeakyReLU(name="Lrelu_"+name)(layer)
        return layer

    def _bulid(self):
        inputs = layers.Input(shape=(self.input_hw[0], self.input_hw[1], 3), name="input")

        feature_model = self._conv(inputs, 32, (3, 3), name="0")
        feature_model = layers.MaxPool2D((2, 2), name="maxpool_0")(feature_model)
        feature_model = self._conv(feature_model, 64, (3, 3), name="1")
        feature_model = layers.MaxPool2D((2, 2), name="maxpool_1")(feature_model)
        feature_model = self._conv(feature_model, 128, (3, 3), name="2")
        feature_model = self._conv(feature_model, 64, (1, 1), name="3")
        feature_model = self._conv(feature_model, 128, (3, 3), name="4")
        feature_model = layers.MaxPool2D((2, 2), name="maxpool_4")(feature_model)
        feature_model = self._conv(feature_model, 256, (3, 3), name="5")
        feature_model = self._conv(feature_model, 128, (1, 1), name="6")
        feature_model = self._conv(feature_model, 256, (3, 3), name="7")
        feature_model = layers.MaxPool2D((2, 2), name="maxpool_7")(feature_model)
        feature_model = self._conv(feature_model, 512, (3, 3), name="8")
        feature_model = self._conv(feature_model, 256, (1, 1), name="9")
        feature_model = self._conv(feature_model, 512, (3, 3), name="10")
        feature_model = self._conv(feature_model, 256, (1, 1), name="11")
        feature_model = self._conv(feature_model, 512, (3, 3), name="12")
        feature_model = layers.MaxPool2D((2, 2), name="maxpool_12")(feature_model)

        reorg = self._conv(feature_model, 2048, (3, 3), name="13")

        feature_model = self._conv(feature_model, 1024, (3, 3), name="14")
        feature_model = self._conv(feature_model, 512, (1, 1), name="15")
        feature_model = self._conv(feature_model, 1024, (3, 3), name="16")
        feature_model = self._conv(feature_model, 512, (1, 1), name="17")
        feature_model = self._conv(feature_model, 1024, (3, 3), name="18")
        feature_model = self._conv(feature_model, 1024, (3, 3), name="19")
        feature_model = self._conv(feature_model, 1024, (3, 3), name="20")

        feature_model = layers.Concatenate(axis=-1, name="concat")([feature_model, reorg])

        feature_model = self._conv(feature_model, 1024, (3, 3), name="21")
        feature_model = layers.Conv2D(self.anchors_size * (5 + self.num_classes), (1, 1), name="conv_22", activation="sigmoid")(feature_model)
        feature_model = layers.Reshape((self.grid_h_cnt, self.grid_w_cnt, self.anchors_size, 5 + self.num_classes))(feature_model)

        feature_model = models.Model(inputs, feature_model)

        feature_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                loss=self._loss,
                                metrics=['accuracy'])
        
        return feature_model

    def _loss(self, y_true, y_pred):
        true_txty = tf.reshape(y_true[..., 0:2], (-1, self.grid_h_cnt * self.grid_h_cnt * self.anchors_size, 2))
        true_twth = tf.reshape(y_true[..., 2:4], (-1, self.grid_h_cnt * self.grid_h_cnt * self.anchors_size, 2))
        true_conf = tf.reshape(y_true[..., 4:5], (-1, self.grid_h_cnt * self.grid_h_cnt * self.anchors_size))
        true_class_probs = tf.reshape(y_true[..., 5:], (-1, self.grid_h_cnt * self.grid_h_cnt * self.anchors_size, self.num_classes))

        pred_txty = tf.reshape(y_pred[..., 0:2], (-1, self.grid_h_cnt * self.grid_h_cnt * self.anchors_size, 2))
        pred_twth = tf.reshape(y_pred[..., 2:4], (-1, self.grid_h_cnt * self.grid_h_cnt * self.anchors_size, 2))
        pred_conf = tf.reshape(y_pred[..., 4:5], (-1, self.grid_h_cnt * self.grid_h_cnt * self.anchors_size))
        pred_class_probs = tf.reshape(y_pred[..., 5:], (-1, self.grid_h_cnt * self.grid_h_cnt * self.anchors_size, self.num_classes))

        bc = losses.BinaryCrossentropy()
        loss_txty = bc(true_txty[true_conf == 1], pred_txty[true_conf == 1], sample_weight=[1])
        loss_twth = bc(true_twth[true_conf == 1], pred_twth[true_conf == 1], sample_weight=[1])
        loss_conf = bc(true_conf, pred_conf, sample_weight=[1])
        loss_class_probs = bc(true_class_probs[true_conf == 1], pred_class_probs[true_conf == 1], sample_weight=[1])
        total_loss = [loss_txty, loss_twth, loss_conf, loss_class_probs]
        # tf.print( total_loss, end='')

        return total_loss

    def train(self):
        pass
    
    def predict(self):
        pass

if __name__ == "__main__" :
    print(tf.test.is_gpu_available())
    print(tf.__version__)