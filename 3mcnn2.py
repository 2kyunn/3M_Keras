import keras
import numpy as np
from keras import models, layers


class CNN(models.Model):
    def __init__(self, input_shape, num_class):
        self.input_shape = input_shape
        self.num_class = num_class
        self.build_model()
        super().__init__(self.x, self.y)
        self.compile()

    def build_model(self):
        num_class = self.num_class
        inp_shape = self.input_shape

        x = layers.Input(inp_shape)
        h = layers.Conv2D(32, (3, 3), activation="relu", input_shape=inp_shape)(x)
        h = layers.Conv2D(64, (3, 3), activation="relu")(h)
        h = layers.MaxPooling2D()(h)
        h = layers.Dropout(0.25)(h)
        h = layers.Flatten()(h)
        z_cl = h

        h = layers.Dense(128, activation="relu")(h)
        h = layers.Dropout(0.5)(h)
        z_fl = h
        y = layers.Dense(num_class, activation="softmax", name="preds")(h)

        self.cl_part = models.Model(x, z_cl)
        self.fl_part = models.Model(x, z_fl)

        self.x, self.y = x, y

    def compile(self):
        models.Model.compile(
            self,
            optimizer="adadelta",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
