# We build the model class here
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential

def CNN():
    model = Sequential()

    model.add(Conv2D(32, (3,3), padding='same', use_bias=False,
                        activation='relu', input_shape = (96,96,1)))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (3,3), padding='same', use_bias=False,
                        activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3), padding='same', use_bias=False,
                        activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3,3), padding='same', use_bias=False,
                        activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(96, (3,3), padding='same', use_bias=False,
                        activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(96, (3,3), padding='same', use_bias=False,
                        activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(128, (3,3), padding='same', use_bias=False,
                        activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3,3), padding='same', use_bias=False,
                        activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(256, (3,3), padding='same', use_bias=False,
                        activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3,3), padding='same', use_bias=False,
                        activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(512, (3,3), padding='same', use_bias=False,
                        activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(512, (3,3), padding='same', use_bias=False,
                        activation='relu'))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(30))

    model.compile(optimizer= tf.keras.optimizers.Adam(0.001),
                    loss='mse',
                    metrics=['mae'])
    return model

if __name__ == "__main__":
    model = CNN()
    model.summary()
