import tensorflow as tf
import tensorflow.keras as keras
import cv2
import numpy as np
def load_mnist(image_size):
    (x_train, y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
    # print(x_train.shape)
    # for image in x_train:
    #     print(image)
    #     image = cv2.resize(image, (1024, 1024))
    #     print(image.shape)
    #     cv2.namedWindow("Image")
    #     cv2.imshow("Image", image)
    #     cv2.waitKey(500)
    #     cv2.destroyAllWindows()

    train_image = [cv2.cvtColor(cv2.resize(img,(image_size,image_size)),cv2.COLOR_GRAY2BGR) for img in x_train]
    test_image = [cv2.cvtColor(cv2.resize(img,(image_size,image_size)),cv2.COLOR_GRAY2BGR) for img in x_test]
    train_image = np.asarray(train_image)
    test_image = np.asarray(test_image)
    # train_label = to_categorical(y_train)
    # test_label = to_categorical(y_test)
    print('finish loading data!')
    return train_image, y_train, test_image, y_test

x, y, x_test, y_test = load_mnist(32)
print(x_test.shape)
model = keras.models.load_model("log/model1")
for img in x_test:
    label = model.predict(img.reshape(-1, 32, 32, 3))
    cv2.namedWindow("Image")
    cv2.imshow("Image", img)
    cv2.waitKey(2000)
    print(np.argmax(label[0]))
    cv2.destroyAllWindows()

