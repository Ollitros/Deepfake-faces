from keras.models import Model
from custom_blocks import *


def Generator(input_shape):
    """
            Autoencoders function creates three models: encoder model (common for two decoders), src decoder (which
        decodes features from common encoder and tries to reconstruct source image), dst decoder (which decodes features
        from common encoder and tries to reconstruct destination image).

        :param input_shape:
        :return: model
    """

    # #######################
    # ## Make encoder
    # #######################

    encoder_inputs = Input(shape=input_shape)
    x = Conv2D(64, kernel_size=5, use_bias=False, padding="same")(encoder_inputs)
    x = conv_block(128)(x)
    x = conv_block(256)(x)
    x = self_attn_block(x, 256)
    x = conv_block(512)(x)
    x = self_attn_block(x, 512)
    x = conv_block(1024)(x)

    activ_map_size = input_shape[0] // 16
    while activ_map_size > 4:
        x = conv_block(1024)(x)
        activ_map_size = activ_map_size // 2

    x = Dense(1024)(Flatten()(x))
    x = Dense(4 * 4 * 1024)(x)
    x = Reshape((4, 4, 1024))(x)
    encoder_output = upscale(512)(x)

    # # #######################
    # # ## Make src_decoder
    # # #######################

    src_inputs = Input(shape=(8, 8, 512))
    x = upscale(256)(src_inputs)
    x = upscale(128)(x)
    x = self_attn_block(x, 128)
    x = upscale(64)(x)
    x = res_block(x, 64)
    x = self_attn_block(x, 64)
    x = upscale(64)(x)

    outputs = []
    activ_map_size = input_shape[0] * 8
    while activ_map_size < 128:
        outputs.append(Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x))
        x = upscale(64)(x)
        x = conv_block(64, strides=1)(x)
        activ_map_size *= 2

    out = Conv2D(3, kernel_size=5, padding='same', activation="sigmoid")(x)
    outputs.append(out)
    src_decoder_output = outputs

    # # #######################
    # # ## Make dst_decoder
    # # #######################

    dst_inputs = Input(shape=(8, 8, 512))
    x = upscale(256)(dst_inputs)
    x = upscale(128)(x)
    x = self_attn_block(x, 128)
    x = upscale(64)(x)
    x = res_block(x, 64)
    x = self_attn_block(x, 64)
    x = upscale(64)(x)

    outputs = []
    activ_map_size = input_shape[0] * 8
    while activ_map_size < 128:
        outputs.append(Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x))
        x = upscale(64)(x)
        x = conv_block(64, strides=1)(x)
        activ_map_size *= 2

    out = Conv2D(3, kernel_size=5, padding='same', activation="sigmoid")(x)
    outputs.append(out)
    dst_decoder_output = outputs

    # Build and compile
    encoder = Model(inputs=encoder_inputs, outputs=encoder_output)
    encoder.compile(loss='mean_squared_error', optimizer='adam')

    src_decoder = Model(inputs=src_inputs, outputs=src_decoder_output)
    src_decoder.compile(loss='mean_squared_error', optimizer='adam')

    dst_decoder = Model(inputs=dst_inputs, outputs=dst_decoder_output)
    dst_decoder.compile(loss='mean_squared_error', optimizer='adam')
    print(encoder.summary())
    print(src_decoder.summary())
    print(dst_decoder.summary())

    return encoder, src_decoder, dst_decoder


def Discriminator(image_shape):

    inputs = Input(shape=image_shape)

    x = d_layer(inputs, 128)
    x = d_layer(x,  256)
    x = d_layer(x,  512)
    x = self_attn_block(x, 512)

    activ_map_size = image_shape[0] // 8
    while activ_map_size > 8:
        x = d_layer(x, 256)
        x = self_attn_block(x, 256)
        activ_map_size = activ_map_size // 2

    out = Conv2D(1, kernel_size=3, padding="same")(x)

    discriminator = Model(inputs, out)
    discriminator.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    print(discriminator.summary())

    return discriminator

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
