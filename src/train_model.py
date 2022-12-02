from keras.datasets import mnist
import matplotlib.pyplot as plt
import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

class Model:
    def __init__(self):
        self.train_data, self.test_data = mnist.load_data()

    def reshape_data(self, x_train, x_test, y_train, y_test):
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        return (x_train, x_test), (y_train, y_test)

    def start_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
            activation='relu',
            input_shape=(28, 28, 1)))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        return model

if __name__ == "__main__":
    d_r = Model()
    (x_train, x_test), (y_train, y_test) = d_r.reshape_data(d_r.train_data[0], d_r.test_data[0], d_r.train_data[1], d_r.test_data[1])
    model = d_r.start_model()

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

    batch_size = 128
    epochs = 10
    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save("trained_model.h5")



