from keras.layers import Input, Conv2D, UpSampling2D, MaxPool2D, BatchNormalization, Dropout, Activation, LeakyReLU, Dense, Reshape, Flatten
from keras.models import Model
from PixelShuffler import PixelShuffler


def Autoencoders(input_shape):
    """
            Autoencoders function creates three models: encoder model (common for two decoders), src decoder (which
        decodes features from common encoder and tries to reconstruct source image), dst decoder (which decodes features
        from common encoder and tries to reconstruct destination image).

        :param input_shape:
        :return: model
    """

    def conv_block(filters):
        def block(x):
            x = Conv2D(filters, kernel_size=3, strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.1)(x)
            return x

        return block

    def upscale(filters):
        def block(x):
            x = Conv2D(filters * 4, kernel_size=3, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.1)(x)
            x = PixelShuffler()(x)
            return x

        return block

    # #######################
    # ## Make encoder
    # #######################

    encoder_inputs = Input(shape=input_shape)
    x = conv_block(256)(encoder_inputs)
    x = conv_block(512)(x)
    x = conv_block(1024)(x)

    encoder_output = upscale(512)(x)

    # # #######################
    # # ## Make src_decoder
    # # #######################

    src_inputs = Input(shape=(32, 32, 512))
    src_decoder_input = upscale(512)(src_inputs)
    x = upscale(256)(src_decoder_input)
    src_decoder_output = Conv2D(3, kernel_size=3, padding='same', activation='sigmoid')(x)

    # # #######################
    # # ## Make dst_decoder
    # # #######################

    dst_inputs = Input(shape=(32, 32, 512))
    dst_decoder_input = upscale(512)(dst_inputs)
    x = upscale(256)(dst_decoder_input)
    dst_decoder_output = Conv2D(3, kernel_size=3, padding='same', activation='sigmoid')(x)

    encoder = Model(inputs=encoder_inputs, outputs=encoder_output)
    encoder.compile(loss='mean_squared_error', optimizer='adam')

    src_decoder = Model(inputs=src_inputs, outputs=src_decoder_output)
    src_decoder.compile(loss='mean_squared_error', optimizer='adam')
    #
    dst_decoder = Model(inputs=dst_inputs, outputs=dst_decoder_output)
    dst_decoder.compile(loss='mean_squared_error', optimizer='adam')
    print(encoder.summary())
    print(src_decoder.summary())
    print(dst_decoder.summary())

    return encoder, src_decoder, dst_decoder


# Autoencoders((256, 256, 3))

# ###################################################################################
# This code just for example to investigate how create such model with gotten weights
# ###################################################################################

# def MergedAutoencoder(X, src_model, dst_model):
#
#     encoder = Model(inputs=src_model.input, outputs=src_model.get_layer('src_encoder_output').output)
#     encoder.compile(loss='mean_squared_error', optimizer='adam')
#     prediction = encoder.predict(X)
#
#     inputs = Input(shape=(25, 25, 512))
#     x = dst_model.get_layer('dst_decoder_input')(inputs)
#     x = dst_model.get_layer('up_sampling2d_3')(x)
#     x = dst_model.get_layer('conv2d_10')(x)
#     x = dst_model.get_layer('up_sampling2d_4')(x)
#     x = dst_model.get_layer('conv2d_11')(x)
#     x = dst_model.get_layer('up_sampling2d_5')(x)
#     x = dst_model.get_layer('decoder_output')(x)
#
#     decoder = Model(inputs=inputs, outputs=x)
#     decoder.compile(loss='mean_squared_error', optimizer='adam')
#     prediction = decoder.predict(prediction)
#
#     return prediction
