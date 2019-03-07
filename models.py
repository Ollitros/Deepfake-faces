from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, MaxPool2D, BatchNormalization
from tensorflow.keras.models import Model
from keras import backend as K


def Autoencoder(input_shape):

    # #######################
    # ## Make src_autoencoder
    # #######################

    # Encoder
    src_encoder_inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(src_encoder_inputs)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    src_encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same', name='src_encoder_output')(x)

    # Decoder
    src_decoder_input = Conv2D(256, (3, 3), activation='relu', padding='same')(src_encoder_output)
    x = UpSampling2D((2, 2))(src_decoder_input)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    src_decoder_output = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    src_autoencoder = Model(inputs=src_encoder_inputs , outputs=src_decoder_output)
    src_autoencoder.compile(loss='mean_squared_error', optimizer='adam')

    # #######################
    # ## Make dst_autoencoder
    # #######################
    dst_inputs = Input(shape=input_shape)

    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(dst_inputs)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    dst_encoder_output = Conv2D(512, (3, 3), activation='relu', padding='same', name='dst_encoder_output')(x)

    # Decoder
    dst_decoder_input = Conv2D(256, (3, 3), activation='relu', padding='same', name='dst_decoder_input')(dst_encoder_output)
    x = UpSampling2D((2, 2))(dst_decoder_input)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoder_output = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='decoder_output')(x)

    dst_autoencoder = Model(inputs=dst_inputs, outputs=decoder_output)
    dst_autoencoder.compile(loss='mean_squared_error', optimizer='adam')

    return src_autoencoder, dst_autoencoder


def MergedAutoencoder(X, src_model, dst_model):

    encoder = Model(inputs=src_model.input, outputs=src_model.get_layer('src_encoder_output').output)
    encoder.compile(loss='mean_squared_error', optimizer='adam')
    prediction = encoder.predict(X)

    inputs = Input(shape=(25, 25, 512))
    x = dst_model.get_layer('dst_decoder_input')(inputs)
    x = dst_model.get_layer('up_sampling2d_3')(x)
    x = dst_model.get_layer('conv2d_10')(x)
    x = dst_model.get_layer('up_sampling2d_4')(x)
    x = dst_model.get_layer('conv2d_11')(x)
    x = dst_model.get_layer('up_sampling2d_5')(x)
    x = dst_model.get_layer('decoder_output')(x)

    decoder = Model(inputs=inputs, outputs=x)
    decoder.compile(loss='mean_squared_error', optimizer='adam')
    prediction = decoder.predict(prediction)

    return prediction
