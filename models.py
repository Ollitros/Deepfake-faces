from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, MaxPool2D, BatchNormalization
from tensorflow.keras.models import Model


def Autoencoder(input_shape):
    inputs = Input(shape=input_shape)

    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same')(x)

    # encoder = Model(inputs=inputs, outputs=encoder_output)

    # Decoder
    decoder_input = Conv2D(256, (3, 3), activation='relu', padding='same')(encoder_output)
    x = UpSampling2D((2, 2))(decoder_input)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoder_output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # decoder = Model(inputs=decoder_input, outputs=decoder_output)

    autoencoder = Model(inputs=inputs, outputs=decoder_output)
    autoencoder.compile(loss='mean_squared_error', optimizer='adam')

    return autoencoder