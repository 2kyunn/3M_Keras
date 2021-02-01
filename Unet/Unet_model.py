import numpy as np
from keras import models, layers
from keras.layers import (
    MaxPooling2D,
    Conv2D,
    concatenate,
    Dropout,
    BatchNormalization,
    UpSampling2D,
    Input,
)
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras import backend


class UNET(models.Model):
    def __init__(self, org_shape, n_class):
        ic = 3 if backend.image_data_format() == "channel_last" else 1
        self.org_shape = org_shape
        self.n_ch = n_class

        def conv(x, n_f, mp_flag=True):
            x = (
                layers.MaxPooling2D((2, 2), strides=2, padding="same")(x)
                if mp_flag
                else x
            )
            x = Conv2D(n_f, (3, 3), padding="same")(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = Dropout(0.5)(x)
            x = Conv2D(n_f, (3, 3), padding="same")(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)

            return x

        def deconv(x, e, n_f):
            x = UpSampling2D((2, 2))(x)
            x = concatenate([x, e], axis=ic)
            x = Conv2D(n_f, (3, 3), padding="same")(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            x = Conv2D(n_f, (3, 3), padding="same")(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
            return x

        original = Input(shape=org_shape)
        c1 = conv(original, 32, mp_flag=False)
        c2 = conv(c1, 64)
        c3 = conv(c2, 128)
        c4 = conv(c3, 256)
        c5 = conv(c4, 512)

        h = deconv(c5, c4, 256)
        h = deconv(h, c3, 128)
        h = deconv(h, c2, 64)
        h = deconv(h, c1, 32)
        decoded = Conv2D(n_class, (1, 1), activation="softmax", padding="same")

        super().__init__(original, decoded)
        super().compile(
            optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )


models.Model.fit_generator
