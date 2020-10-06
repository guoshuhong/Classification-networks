from tensorflow.python.keras.utils.np_utils import to_categorical

from net import  DenseNet
import tensorflow as tf
import tensorflow.keras as keras
import cv2
import numpy as np

def load_mnist(image_size):
    (x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
    train_image = [cv2.cvtColor(cv2.resize(img,(image_size,image_size)),cv2.COLOR_GRAY2BGR) for img in x_train]
    test_image = [cv2.cvtColor(cv2.resize(img,(image_size,image_size)),cv2.COLOR_GRAY2BGR) for img in x_test]
    train_image = np.asarray(train_image)
    test_image = np.asarray(test_image)
    # train_label = to_categorical(y_train)
    # test_label = to_categorical(y_test)
    print('finish loading data!')
    return train_image, y_train, test_image, y_test

x, y, x_test, y_test = load_mnist(56)


# 类别onehot编码
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
y = encoder.fit_transform(y)
y_test = encoder.fit_transform(y_test)
print(y)
# x = x.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)
print(y.shape, y_test.shape)
model = DenseNet.DenseNet121(weights=None, classes=10)
model.summary()
optimizer = tf.keras.optimizers.Adam(0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
log_dir = ""
# 使用命令tensorboard --logdir=. 即可查看tensorboard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,)
history = model.fit(x, y, epochs=10, batch_size=8,
                    validation_data=(x_test, y_test), callbacks=[tensorboard_callback])
print(history)