import numpy as np
import keras
from keras import models, layers
from keras import backend  # make functions keras does not support
from keras import datasets


class CNN_seq(models.Sequential):
    def __init__(self, input_shape, Nout):
        super().__init__()
        ### convolution --> image feature detection ###
        self.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))

        self.add(layers.Conv2D(64, (3, 3), activation="relu"))
        self.add(layers.MaxPooling2D((2, 2)))
        self.add(layers.Dropout(0.25))

        self.add(layers.Flatten())

        ### fully connected layers -> classification ###
        self.add(layers.Dense(128, activation="relu"))
        self.add(layers.Dropout(0.5))
        self.add(layers.Dense(Nout, activation="softmax"))

        self.compile(
            keras.optimizers.Adadelta(),
            keras.losses.categorical_crossentropy,
            metrics=["accuracy"],
        )


########## DATA PREPROCESSING ##########
# MNIST -> no channel --> preprocess ##from keras import datasets


class DATA:
    def __init__(self):
        Nout = 10
        (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data(
            path="C:\\Users\\Jungyun\\Desktop\\3mkeras\\datasets\\mnist.npz"
        )

        img_rows, img_cols = x_train.shape[1:]

        if backend.image_data_format() == "chanenl_first":
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)

        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype("float32")
        x_test = x_test.astype("float32")
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        y_train = keras.utils.to_categorical(y_train, Nout)
        y_test = keras.utils.to_categorical(y_test, Nout)

        self.input_shape = input_shape
        self.num_class = Nout
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test


def main():
    batch_size = 128
    epochs = 10

    data = DATA()
    model = CNN_seq(data.input_shape, data.num_class)
    history = model.fit(
        data.x_train,
        data.y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2,
    )
    score = model.evaluate(data.x_test, data.y_test)
    print()
    print("Test Loss: ", score[0])
    print("Test Accuracy: ", score[1])


if __name__ == "__main__":
    main()
