from keras.models import Model
from network.custom_blocks import *
from keras.optimizers import Adam
from network.losses import *


class GanModel:

    def __init__(self, input_shape, image_shape):
        self.input_shape = input_shape
        self.image_shape = image_shape
        self.lrD = 2e-4
        self.lrG = 1e-4

        # Define networks
        self.encoder, self.src_decoder, self.dst_decoder = self.Generator(input_shape=self.input_shape)

        # Create discriminators
        self.src_discriminator = self.Discriminator(image_shape=self.image_shape)
        self.dst_discriminator = self.Discriminator(image_shape=self.image_shape)

        # Combining two separate models into one. Required creating Input layer.
        # Create common encoder
        self.encoder_input = Input(shape=input_shape)
        self.encode = self.encoder(self.encoder_input)

        # Create generators
        self.src_decode = self.src_decoder(self.encode)
        self.dst_decode = self.dst_decoder(self.encode)

        self.src_gen = Model(self.encoder_input, self.src_decode)
        self.dst_gen = Model(self.encoder_input, self.dst_decode)

        # Define variables
        self.distorted_src, self.fake_src, self.mask_src, self.path_src, self.path_mask_src, \
        self.path_abgr_src, self.path_bgr_src = self.define_variables(generator=self.src_gen)
        self.distorted_dst, self.fake_dst, self.mask_dst, self.path_dst, self.path_mask_dst, \
        self.path_abgr_dst, self.path_bgr_dst = self.define_variables(generator=self.dst_gen)

        self.real_src = Input(shape=self.input_shape)
        self.real_dst = Input(shape=self.input_shape)

    def Generator(self, input_shape):
        """
                Generator function creates three models: encoder model (common for two decoders), src decoder (which
            decodes features from common encoder and tries to reconstruct source image), dst decoder (which decodes features
            from common encoder and tries to reconstruct destination image).

            :param input_shape:
            :return: model
        """

        encoder_inputs = src_inputs = dst_inputs = encoder_output = src_decoder_output = dst_decoder_output = None

        if self.input_shape[0] == input_shape[1] == 64:

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

            outputs = []
            activ_map_size = input_shape[0] * 8
            while activ_map_size < 128:
                outputs.append(Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x))
                x = upscale(64)(x)
                x = conv_block(64, strides=1)(x)
                activ_map_size *= 2

            alpha = Conv2D(1, kernel_size=5, padding='same', activation="sigmoid")(x)
            bgr = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x)
            out = concatenate([alpha, bgr])
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

            outputs = []
            activ_map_size = input_shape[0] * 8
            while activ_map_size < 128:
                outputs.append(Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x))
                x = upscale(64)(x)
                x = conv_block(64, strides=1)(x)
                activ_map_size *= 2

            alpha = Conv2D(1, kernel_size=5, padding='same', activation="sigmoid")(x)
            bgr = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x)
            out = concatenate([alpha, bgr])
            outputs.append(out)
            dst_decoder_output = outputs

        if self.input_shape[0] == input_shape[1] == 128:

            # #######################
            # ## Make encoder
            # #######################

            encoder_inputs = Input(shape=input_shape)
            x = Conv2D(64, kernel_size=5, use_bias=False, padding="same")(encoder_inputs)
            x = conv_block(128)(x)
            x = conv_block(256)(x)
            x = self_attn_block(x, 256)
            x = conv_block(256)(x)
            x = self_attn_block(x, 256)
            x = conv_block(512)(x)
            x = self_attn_block(x, 512)
            x = conv_block(1024)(x)

            activ_map_size = input_shape[0] // 32
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
            x = self_attn_block(x, 64)

            outputs = []
            activ_map_size = input_shape[0] * 8
            while activ_map_size < 128:
                outputs.append(Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x))
                x = upscale(64)(x)
                x = conv_block(64, strides=1)(x)
                activ_map_size *= 2

            alpha = Conv2D(1, kernel_size=5, padding='same', activation="sigmoid")(x)
            bgr = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x)
            out = concatenate([alpha, bgr])
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
            x = self_attn_block(x, 64)

            outputs = []
            activ_map_size = input_shape[0] * 8
            while activ_map_size < 128:
                outputs.append(Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x))
                x = upscale(64)(x)
                x = conv_block(64, strides=1)(x)
                activ_map_size *= 2

            alpha = Conv2D(1, kernel_size=5, padding='same', activation="sigmoid")(x)
            bgr = Conv2D(3, kernel_size=5, padding='same', activation="tanh")(x)
            out = concatenate([alpha, bgr])
            outputs.append(out)
            dst_decoder_output = outputs

        # Build and compile
        encoder = Model(inputs=encoder_inputs, outputs=encoder_output)
        src_decoder = Model(inputs=src_inputs, outputs=src_decoder_output)
        dst_decoder = Model(inputs=dst_inputs, outputs=dst_decoder_output)

        print(encoder.summary())
        print(src_decoder.summary())
        print(dst_decoder.summary())

        return encoder, src_decoder, dst_decoder

    def Discriminator(self, image_shape):

        inputs = Input(shape=image_shape)

        x = dis_layer(inputs, 128)
        x = dis_layer(x, 256)
        x = dis_layer(x, 512)
        x = self_attn_block(x, 512)

        activ_map_size = image_shape[0] // 8
        while activ_map_size > 8:
            x = dis_layer(x, 256)
            x = self_attn_block(x, 256)
            activ_map_size = activ_map_size // 2

        out = Conv2D(1, kernel_size=3, padding="same")(x)

        discriminator = Model(inputs, out)
        print(discriminator.summary())

        return discriminator

    @staticmethod
    def define_variables(generator):
        distorted_input = generator.inputs[0]
        fake_output = generator.outputs[-1]
        alpha = Lambda(lambda x: x[:, :, :, :1])(fake_output)
        bgr = Lambda(lambda x: x[:, :, :, 1:])(fake_output)

        masked_fake_output = alpha * bgr + (1 - alpha) * distorted_input

        fn_generate = K.function([distorted_input], [masked_fake_output])   # Return almost like input
        fn_mask = K.function([distorted_input], [concatenate([alpha, alpha, alpha])]) # Return black image
        fn_abgr = K.function([distorted_input], [concatenate([alpha, bgr])]) # Return 4-channels maksed image
        fn_bgr = K.function([distorted_input], [bgr])   # Return input + mask

        return distorted_input, fake_output, alpha, fn_generate, fn_mask, fn_abgr, fn_bgr

    def build_train_functions(self):

        # Adversarial loss
        loss_src_dis, loss_adv_src_gen = adversarial_loss(self.src_discriminator, self.real_src, self.fake_src, self.distorted_src)
        loss_dst_dis, loss_adv_dst_gen = adversarial_loss(self.dst_discriminator, self.real_dst, self.fake_dst, self.distorted_dst)
        loss_src_gen = loss_adv_src_gen
        loss_dst_gen = loss_adv_dst_gen

        # Alpha mask loss
        loss_src_gen += 1e-2 * K.mean(K.abs(self.mask_src))
        loss_dst_gen += 1e-2 * K.mean(K.abs(self.mask_dst))

        # Alpha mask total variation loss
        loss_src_gen += 0.1 * K.mean(first_order(self.mask_src, axis=1))
        loss_src_gen += 0.1 * K.mean(first_order(self.mask_src, axis=2))
        loss_dst_gen += 0.1 * K.mean(first_order(self.mask_dst, axis=1))
        loss_dst_gen += 0.1 * K.mean(first_order(self.mask_dst, axis=2))

        # L2 weight decay
        # https://github.com/keras-team/keras/issues/2662
        for loss_tensor in self.src_gen.losses:
            loss_src_gen += loss_tensor
        for loss_tensor in self.dst_gen.losses:
            loss_dst_gen += loss_tensor
        for loss_tensor in self.src_discriminator.losses:
            loss_src_dis += loss_tensor
        for loss_tensor in self.dst_discriminator.losses:
            loss_dst_dis += loss_tensor

        weights_src_dis = self.src_discriminator.trainable_weights
        weights_src_gen = self.src_gen.trainable_weights
        weights_dst_dis = self.dst_discriminator.trainable_weights
        weights_dst_gen = self.dst_gen.trainable_weights

        # Define training functions
        lr_factor = 1
        training_updates = Adam(lr=self.lrD * lr_factor, beta_1=0.5).get_updates(weights_src_dis, [], loss_src_dis)
        self.net_src_dis_train = K.function([self.distorted_src, self.real_src], [loss_src_dis], training_updates)
        training_updates = Adam(lr=self.lrG * lr_factor, beta_1=0.5).get_updates(weights_src_gen, [], loss_src_gen)
        self.net_src_gen_train = K.function([self.distorted_src, self.real_src], [loss_src_gen, loss_adv_src_gen], training_updates)

        training_updates = Adam(lr=self.lrD * lr_factor, beta_1=0.5).get_updates(weights_dst_dis, [], loss_dst_dis)
        self.net_dst_dis_train = K.function([self.distorted_dst, self.real_dst], [loss_dst_dis], training_updates)
        training_updates = Adam(lr=self.lrG * lr_factor, beta_1=0.5).get_updates(weights_dst_gen, [], loss_dst_gen)
        self.net_dst_gen_train = K.function([self.distorted_dst, self.real_dst], [loss_dst_gen, loss_adv_dst_gen], training_updates)

    def load_weights(self, path="data/models"):
        self.encoder.load_weights("{path}/encoder.h5".format(path=path))
        self.src_decoder.load_weights("{path}/decoder_A.h5".format(path=path))
        self.dst_decoder.load_weights("{path}/decoder_B.h5".format(path=path))
        self.src_discriminator.load_weights("{path}/netDA.h5".format(path=path))
        self.dst_discriminator.load_weights("{path}/netDB.h5".format(path=path))
        print("Model weights files are successfully loaded.")

    def save_weights(self, path="data/models"):
        self.encoder.save("{path}/encoder.h5".format(path=path))
        self.src_decoder.save("{path}/decoder_A.h5".format(path=path))
        self.dst_decoder.save("{path}/decoder_B.h5".format(path=path))
        self.src_discriminator.save("{path}/netDA.h5".format(path=path))
        self.dst_discriminator.save("{path}/netDB.h5".format(path=path))
        print("Model weights files have been saved to {path}.".format(path=path))

    def train_generators(self, X, Y):

        err_src_gen = self.net_src_gen_train([X, X])
        err_dst_gen = self.net_dst_gen_train([Y, Y])

        return err_src_gen, err_dst_gen

    def train_discriminators(self, X, Y):

        err_src_dis = self.net_src_dis_train([X, X])
        err_dst_dis = self.net_dst_dis_train([Y, Y])

        return err_src_dis, err_dst_dis

    def transform_src_to_dst(self, img):
        return self.path_abgr_dst([[img]])

    def transform_dst_to_src(self, img):
        return self.path_abgr_src([[img]])


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
