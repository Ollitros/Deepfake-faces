from keras.layers import Input, Conv2D, UpSampling2D, MaxPool2D, BatchNormalization, LeakyReLU, Dropout, Activation
from keras.models import Model


def Autoencoders(input_shape):

    def conv_block(inputs, filters, kernel_size, padding, activation):
        x = Conv2D(filters, kernel_size, padding=padding)(inputs)
        x = BatchNormalization()(x)
        x = Activation(activation=activation)(x)

        return x

    # #######################
    # ## Make encoder
    # #######################

    encoder_inputs = Input(shape=input_shape)
    x = conv_block(encoder_inputs, 128, (3, 3), padding='same', activation='relu')
    x = Dropout(0.25)(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = conv_block(x, 256, (3, 3), padding='same', activation='relu')
    x = Dropout(0.25)(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = conv_block(x, 512, (3, 3), padding='same', activation='relu')
    x = Dropout(0.25)(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    encoder_output = conv_block(x, 1024, (3, 3), padding='same', activation='relu')

    # #######################
    # ## Make src_decoder
    # #######################

    src_inputs = Input(shape=(25, 25, 1024))
    src_decoder_input = conv_block(src_inputs, 1024, (3, 3), padding='same', activation='relu')
    src_decoder_input = Dropout(0.25)(src_decoder_input)
    x = UpSampling2D((2, 2))(src_decoder_input)
    x = conv_block(x, 512, (3, 3), padding='same', activation='relu')
    x = Dropout(0.25)(x)
    x = UpSampling2D((2, 2))(x)
    x = conv_block(x, 256, (3, 3), padding='same', activation='relu')
    x = Dropout(0.25)(x)
    x = UpSampling2D((2, 2))(x)
    src_decoder_output = conv_block(x, 3, (3, 3), activation='sigmoid', padding='same')

    # #######################
    # ## Make dst_decoder
    # #######################

    dst_inputs = Input(shape=(25, 25, 1024))
    dst_decoder_input = conv_block(dst_inputs, 1024, (3, 3), padding='same', activation='relu')
    dst_decoder_input = Dropout(0.25)(dst_decoder_input)
    x = UpSampling2D((2, 2))(dst_decoder_input)
    x = conv_block(x, 512, (3, 3), padding='same', activation='relu')
    x = Dropout(0.25)(x)
    x = UpSampling2D((2, 2))(x)
    x = conv_block(x, 256, (3, 3), padding='same', activation='relu')
    x = Dropout(0.25)(x)
    x = UpSampling2D((2, 2))(x)
    dst_decoder_output = conv_block(x, 3, (3, 3), activation='sigmoid', padding='same')

    encoder = Model(inputs=encoder_inputs, outputs=encoder_output)
    encoder.compile(loss='mean_squared_error', optimizer='adam')

    src_decoder = Model(inputs=src_inputs, outputs=src_decoder_output)
    src_decoder.compile(loss='mean_squared_error', optimizer='adam')

    dst_decoder = Model(inputs=dst_inputs, outputs=dst_decoder_output)
    dst_decoder.compile(loss='mean_squared_error', optimizer='adam')

    return encoder, src_decoder, dst_decoder


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
