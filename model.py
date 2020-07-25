from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers


def vgg_style(input_tensor):
    """
    The original feature extraction structure from CRNN paper.
    Related paper: https://ieeexplore.ieee.org/abstract/document/7801919
    """
    x = layers.Conv2D(64, 3, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D(pool_size=2, padding='same')(x)

    x = layers.Conv2D(128, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPool2D(pool_size=2, padding='same')(x)

    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(256, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPool2D(pool_size=2, strides=(2, 1), padding='same')(x)

    x = layers.Conv2D(512, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPool2D(pool_size=2, strides=(2, 1), padding='same')(x)

    x = layers.Conv2D(512, 2, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    return x


def build_model(num_classes, image_width=None, channels=1, training=True):
    """build CNN-RNN model"""

    img_input = keras.Input(shape=(32, image_width, channels))
    x = vgg_style(img_input)
    #if training:
    #    x = layers.Reshape((63, 512))(x)
    #else:
    #    x = x[0, 0, :, :]
    x = layers.Conv2D(filters=num_classes, kernel_size=(1,1), padding='valid')(x)
    x = x[:,0,:,:]
    #x = layers.Dense(units=num_classes,kernel_regularizer=regularizers.l2(1e-4))(x)
    return keras.Model(inputs=img_input, outputs=x, name='CRNN')
