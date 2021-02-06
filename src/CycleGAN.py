from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from utils import ReflectionPadding2D



class Gan():
    def __init__(self):
        img_shape = (opt.img_rows, opt.img_cols, opt.channels)
        

    def build_discriminator(img_shape):
        'Define the Discriminator model'
        # initialization weight
        init = RandomNormal(stddev=0.02)
        # input_image
        in_image = Input(shape=img_shape)

        def disc_layer(in_image, out_channels, strides=(2,2), instance_norm=True, initializer=init):
            'Layer for building Discriminator'
            d = Conv2D(out_channels, kernel_size=(4,4), strides=strides, padding='same', kernel_initializer=initializer)(in_image)
            if instance_norm:
                d = InstanceNormalization(axis=-1)(d)
            d = LeakyReLU(alpha=0.2)(d)
            return d

        # convolutions layers
        d = disc_layer(in_image, 64, instance_norm=False)
        d = disc_layer(d, 128)
        d = disc_layer(d, 256)
        d = disc_layer(d, 512)
        d = disc_layer(d, 512, strides=(1,1))

        # output layer
        out = Conv2D(1, 4, padding='same', kernel_initializer=init)(d)

        # define model
        model = Model(in_image, out)
        return model

    def build_generator(img_shape, n_resnet=9):
        'Define the Generator model'
        # initialization weight
        init = RandomNormal(stddev=0.02)
        # input_image
        in_image = Input(shape=img_shape)

        def resnet_block(n_filters, input_layer, initializer=init):
            'Residual Connection block for building generator'

            # first layer
            rb = Conv2D(filters=n_filters, kernel_size=3, padding='same', kernel_initializer=initializer)(input_layer)
            rb = InstanceNormalization(axis=-1)(rb)
            rb = Activation('relu')(rb)

            # second layer
            rb = Conv2D(filters=n_filters, kernel_size=3, padding='same', kernel_initializer=initializer)(rb)
            rb = InstanceNormalization(axis=-1)(rb)

            # residual connection 
            rb = Concatenate()([rb, input_layer])
            return rb

        def main_block(input_layer, in_features=64, downsampling=True, initializer=init):
            'Downsampling or Upsampling block'
            if downsampling == True:
                out_features = in_features*2
                g = Conv2D(out_features, kernel_size=3, strides=(2,2), padding='same', kernel_initializer=initializer)(input_layer)
            elif downsampling == False:
                out_features = in_features//2
                g = UpSampling2D(size=2, interpolation='bilinear')(input_layer)
                g = ReflectionPadding2D()(g)
                g = Conv2D(out_features, kernel_size=3, strides=1, padding='valid', kernel_initializer=initializer)(g)
            
            g = InstanceNormalization(axis=-1)(g)
            g = Activation('relu')(g)
            return g

        # c7s1-64
        g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
        g = InstanceNormalization(axis=-1)(g)
        g = Activation('relu')(g)
    
        # d128     
        g = main_block(input_layer=g, in_features=64, downsampling=True)
        # d256
        g = main_block(input_layer=g, in_features=128, downsampling=True)

        # R256
        for _ in range(n_resnet):
            g = resnet_block(256, g)

        # u128
        g = main_block(input_layer=g, in_features=256, downsampling=False)
        # u64
        g = main_block(input_layer=g, in_features=128, downsampling=False)

        # c7s1-3
        g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
        g = InstanceNormalization(axis=-1)(g)
        out_image = Activation('tanh')(g)
    
        model = Model(in_image, out_image)
        return model