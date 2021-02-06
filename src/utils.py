import tensorflow as tf
import keras.backend as K
from data_loader import DataLoader
from keras.engine.topology import Layer
from keras.engine import InputSpec




class ReflectionPadding2D(Layer):
    def __init__(self, **kwargs):
        self.padding = tuple((1,1))
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__()

    def get_output_shape_for(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')




def update_learning_rate(model, decay):
     updated_lr = K.get_value(model.optimizer.lr) - decay
     print(updated_lr)
     if updated_lr < 0:
         updated_lr = 1e-9
     K.set_value(model.optimizer.lr, updated_lr)
 
def calculate_linear_lr_decay():
    max_imgs_count = DataLoader.check_min_count()
    
    updates_per_epoch_D = 2 * max_imgs_count
    updates_per_epoch_G = max_imgs_count
    denominator_D = (opt.epochs - opt.decay_epoch) * updates_per_epoch_D
    denominator_G = (opt.epochs - opt.decay_epoch) * updates_per_epoch_G
    decay_D = opt.lr / denominator_D
    decay_G = opt.lr / denominator_G
    return decay_D, decay_G


def update_learning_rate_2(model, epoch):
    updated_lr = self.lr * (1 - 1 / (opt.epochs - opt.decay_epoch) * (epoch - opt.decay_epoch))
    if updated_lr < 0:
        updated_lr = 1e-6
    #print(updated_lr)
    K.set_value(model.optimizer.lr, updated_lr)