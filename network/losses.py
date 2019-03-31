from keras.layers import Lambda, concatenate
from tensorflow.contrib.distributions import Beta
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import keras.backend as K
import tensorflow as tf


def first_order(x, axis=1):
    img_nrows = x.shape[1]
    img_ncols = x.shape[2]
    if axis == 1:
        return K.abs(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    elif axis == 2:
        return K.abs(x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    else:
        return None


def calc_loss(pred, target, loss='l2'):
    if loss.lower() == "l2":
        return K.mean(K.square(pred - target))
    elif loss.lower() == "l1":
        return K.mean(K.abs(pred - target))
    elif loss.lower() == "cross_entropy":
        return -K.mean(K.log(pred + K.epsilon()) * target + K.log(1 - pred + K.epsilon()) * (1 - target))
    else:
        raise ValueError(f'Recieve an unknown loss type: {loss}.')


def adversarial_loss(netD, real, fake_abgr, distorted, gan_training="mixup_LSGAN"):
    weights = {}
    weights['w_D'] = 0.1  # Discriminator
    weights['w_recon'] = 1.  # L1 reconstruction loss
    weights['w_edge'] = 0.1  # edge loss
    weights['w_eyes'] = 30.  # reconstruction and edge loss on eyes area
    weights['w_pl'] = (0.01, 0.1, 0.3, 0.1)  # perceptual loss (0.003, 0.03, 0.3, 0.3)
    alpha = Lambda(lambda x: x[:, :, :, :1])(fake_abgr)
    fake_bgr = Lambda(lambda x: x[:, :, :, 1:])(fake_abgr)
    fake = alpha * fake_bgr + (1 - alpha) * distorted

    if gan_training == "mixup_LSGAN":
        dist = Beta(0.2, 0.2)
        lam = dist.sample()
        mixup = lam * concatenate([real, distorted]) + (1 - lam) * concatenate([fake, distorted])
        pred_fake = netD(concatenate([fake, distorted]))
        pred_mixup = netD(mixup)
        loss_D = calc_loss(pred_mixup, lam * K.ones_like(pred_mixup), "l2")
        loss_G = weights['w_D'] * calc_loss(pred_fake, K.ones_like(pred_fake), "l2")
        mixup2 = lam * concatenate([real, distorted]) + (1 - lam) * concatenate([fake_bgr, distorted])
        pred_fake_bgr = netD(concatenate([fake_bgr, distorted]))
        pred_mixup2 = netD(mixup2)
        loss_D += calc_loss(pred_mixup2, lam * K.ones_like(pred_mixup2), "l2")
        loss_G += weights['w_D'] * calc_loss(pred_fake_bgr, K.ones_like(pred_fake_bgr), "l2")
    elif gan_training == "relativistic_avg_LSGAN":
        real_pred = netD(concatenate([real, distorted]))
        fake_pred = netD(concatenate([fake, distorted]))
        loss_D = K.mean(K.square(real_pred - K.ones_like(fake_pred))) / 2
        loss_D += K.mean(K.square(fake_pred - K.zeros_like(fake_pred))) / 2
        loss_G = weights['w_D'] * K.mean(K.square(fake_pred - K.ones_like(fake_pred)))

        fake_pred2 = netD(concatenate([fake_bgr, distorted]))
        loss_D += K.mean(K.square(real_pred - K.mean(fake_pred2, axis=0) - K.ones_like(fake_pred2))) / 2
        loss_D += K.mean(K.square(fake_pred2 - K.mean(real_pred, axis=0) - K.zeros_like(fake_pred2))) / 2
        loss_G += weights['w_D'] * K.mean(
            K.square(real_pred - K.mean(fake_pred2, axis=0) - K.zeros_like(fake_pred2))) / 2
        loss_G += weights['w_D'] * K.mean(
            K.square(fake_pred2 - K.mean(real_pred, axis=0) - K.ones_like(fake_pred2))) / 2
    else:
        raise ValueError("Receive an unknown GAN training method: {gan_training}")
    return loss_D, loss_G
