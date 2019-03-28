from keras.layers import *
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from custom_layers import PixelShuffler, Scale
import keras.backend as K

conv_init = 'he_normal'
w_l2 = 1e-4


def conv_block(filters, strides=2):
    def block(x):
        x = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(x)
        x = InstanceNormalization()(x)
        x = LeakyReLU(0.2)(x)
        return x
    return block


def d_layer(layer_input, filters, f_size=4):
    """Discriminator layer"""
    x = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = InstanceNormalization()(x)

    return x


def upscale(filters):
    def block(x):
        x = Conv2D(filters * 4, kernel_size=3, padding='same')(x)
        x = InstanceNormalization()(x)
        x = LeakyReLU(0.2)(x)
        x = PixelShuffler()(x)
        return x
    return block


def self_attn_block(inp, nc, squeeze_factor=8):
    '''
    Code borrows from https://github.com/taki0112/Self-Attention-GAN-Tensorflow
    '''
    assert nc // squeeze_factor > 0, f"Input channels must be >= {squeeze_factor}, recieved nc={nc}"
    x = inp
    shape_x = x.get_shape().as_list()

    f = Conv2D(nc // squeeze_factor, 1, kernel_regularizer=regularizers.l2(w_l2))(x)
    g = Conv2D(nc // squeeze_factor, 1, kernel_regularizer=regularizers.l2(w_l2))(x)
    h = Conv2D(nc, 1, kernel_regularizer=regularizers.l2(w_l2))(x)

    shape_f = f.get_shape().as_list()
    shape_g = g.get_shape().as_list()
    shape_h = h.get_shape().as_list()
    flat_f = Reshape((-1, shape_f[-1]))(f)
    flat_g = Reshape((-1, shape_g[-1]))(g)
    flat_h = Reshape((-1, shape_h[-1]))(h)

    s = Lambda(lambda x: K.batch_dot(x[0], Permute((2, 1))(x[1])))([flat_g, flat_f])

    beta = Softmax(axis=-1)(s)
    o = Lambda(lambda x: K.batch_dot(x[0], x[1]))([beta, flat_h])
    o = Reshape(shape_x[1:])(o)
    o = Scale()(o)

    out = add([o, inp])

    return out


def res_block(input_tensor, filters, w_l2=w_l2):

    x = input_tensor
    x = Conv2D(filters, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2),
               kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = InstanceNormalization()(x)
    x = Conv2D(filters, kernel_size=3, kernel_regularizer=regularizers.l2(w_l2),
               kernel_initializer=conv_init, use_bias=False, padding="same")(x)
    x = add([x, input_tensor])
    x = LeakyReLU(alpha=0.2)(x)
    x = InstanceNormalization()(x)

    return x