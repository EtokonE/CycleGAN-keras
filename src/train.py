import argparse

from data_loader import DataLoader
from CycleGAN import Gan
from utils import *

from keras.optimizers import Adam
from keras.models import load_model
from keras.models import Input
from keras.models import Model
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_rows", type=int, default=128, help="size of image height")
    parser.add_argument("--img_cols", type=int, default=128, help="size of image width")
    parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--use_lr_decay', type=bool, default=True, help='if True -> the model usus learning rate decay')
    parser.add_argument('--decay_epoch', type=int, default=101, help='epoch after wich start lr decay')
    parser.add_argument('--dataset_name', type=str, default='monet2photo', help='name of the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='size of the batch')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--beta_1', type=float, default=0.5, help='adam: beta_1')
    parser.add_argument('--beta_2', type=float, default=0.999, help='adam: beta_2')
    parser.add_argument('--lambda_validation', type=float, default=1.0, help='meaning loss weight: are the images created by generator corresponding to the discriminator')
    parser.add_argument('--lambda_reconstr', type=float, default=10.0, help='meaning loss weight: if we use two generators one after the other (in both directions), will the original image work')
    parser.add_argument('--lambda_id', type=float, default=5.0, help='meaning loss weight: if we apply each generator to images from target area, will the image remain unchanged?')
    parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
    parser.add_argument('--sample_interval', type=int, default=300, help='interval to save models and val images')
    parser.add_argument('--increase_generator_iteration', type=bool, default=False, help='increase the number of generator iterations in train loop')
    parser.add_argument('--generator_iterations', type=bool, default=3, help='number of generator iterations in train loop')
    return parser.parse_args()

def sample_images(epoch, batch):
    os.makedirs('../images/%s' % opt.dataset_name, exist_ok=True)
    r, c = 2, 3
    imgs_A = data_loader.load_dataset(domain="A", batch_size=1, test=True)
    imgs_B = data_loader.load_dataset(domain="B", batch_size=1, test=True)
    # Translate images to the other domain
    fake_B = g_AtoB.predict(imgs_A)
    fake_A = g_BtoA.predict(imgs_B)
    # Translate back to original domain
    reconstr_A = g_BtoA.predict(fake_B)
    reconstr_B = g_AtoB.predict(fake_A)
    gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    titles = ['Original', 'Translated', 'Reconstructed']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[j])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("../images/%s/%d_%d.png" % (opt.dataset_name, epoch, batch))
    plt.close()


if __name__ == '__main__':
    
    #__________OPTIONS______________
    opt = create_parser()
    print(f'Your model has started training with the following options: {opt}')
    #__________DATALOADER___________
    data_loader = DataLoader(dataset=opt.dataset_name,
                                  image_shape=(opt.img_rows, opt.img_cols), batch_size=opt.batch_size)

    #_________OPTIMIZER_____________
    optimizer = Adam(lr=opt.lr, beta_1=opt.beta_1, beta_2=opt.beta_2)

    os.makedirs('../images/%s' % opt.dataset_name, exist_ok=True)
    os.makedirs('../models/%s' % opt.dataset_name, exist_ok=True)

    # Calculate output shape of Discriminator (PatchGAN)
    patch = int(opt.img_rows / 2**4)
    disc_patch = (patch, patch, 1)
    image_shape=(opt.img_rows, opt.img_cols, opt.input_nc)
    # build and compile discriminator: A -> [real/fake]
    d_model_A = Gan.build_discriminator(image_shape)
    #d_model_A = load_model('../models/discriminatorA.h5',custom_objects={'InstanceNormalization': InstanceNormalization}, compile=False)
    d_model_A.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])


    # build and compile discriminator: B -> [real/fake]
    d_model_B = Gan.build_discriminator(image_shape)
    #d_model_B = load_model('../models/discriminatorB.h5',custom_objects={'InstanceNormalization': InstanceNormalization}, compile=False)
    d_model_B.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])


    # build generator: A -> B
    g_AtoB = Gan.build_generator(image_shape)
    #g_AtoB = load_model('../models/generatorA2B.h5',custom_objects={'InstanceNormalization': InstanceNormalization, 'ReflectionPadding2D':ReflectionPadding2D()}), compile=False) 

    # build generator: B -> A
    g_BtoA = Gan.build_generator(image_shape)
    #g_BtoA = load_model('../models/generatorB2A.h5',custom_objects={'InstanceNormalization': InstanceNormalization, 'ReflectionPadding2D':ReflectionPadding2D()}), compile=False)

    # define input images for both domains
    img_A = Input(shape=image_shape)
    img_B = Input(shape=image_shape)
                  
    # forward cycle
    fake_B = g_AtoB(img_A)
    fake_A = g_BtoA(img_B)
    # backward cycle (--lambda_reconstr)
    back_to_A = g_BtoA(fake_B)
    back_to_B = g_AtoB(fake_A)
    # Identity mapping of images (--lambda_id)
    img_A_id = g_BtoA(img_A)
    img_B_id = g_AtoB(img_B)
    # For the combined model we will only train the generators
    d_model_A.trainable = False
    d_model_B.trainable = False
    # Discriminators determines validity of translated images (--lambda_validation)
    valid_A = d_model_A(fake_A)
    valid_B = d_model_B(fake_B)

    # define a composite model for updating generators by adversarial and cycle loss
    composite_model = Model(inputs=[img_A, img_B], outputs=[valid_A, valid_B, back_to_A, back_to_B, img_A_id, img_B_id])
    # compile model with weighting of least squares loss and L1 loss
    composite_model.compile(loss=['mse', 'mse',
                                  'mae', 'mae', 
                                  'mae', 'mae'],
                                 loss_weights=[opt.lambda_validation,
                                               opt.lambda_validation,
                                               opt.lambda_reconstr,
                                               opt.lambda_reconstr,
                                               opt.lambda_id,
                                               opt.lambda_id
                                               ], 
                                 optimizer = optimizer)



    #______________________TRAINING_____________________ 


    start_time = datetime.datetime.now()
    # Adversarial loss ground truths
    valid = np.ones((opt.batch_size,) + disc_patch)
    fake = np.zeros((opt.batch_size,) + disc_patch)
    # define image pool for fake images
    poolA, poolB = list(), list()

    # Train loop
    for epoch in range(opt.n_epochs):
        for batch, (img_A, img_B) in enumerate(data_loader.load_batch(opt.batch_size)):
        
            #_______________TRAIN__DISCRIMINATORS___________________
            # translate images to the other domain
            fake_B = g_AtoB.predict(img_A)
            fake_A = g_BtoA.predict(img_B)
            # update fakes from pool
            X_fake_A = data_loader.update_image_pool(poolA, fake_A)
            X_fake_B = data_loader.update_image_pool(poolB, fake_B)
            # train Discriminator A
            d_model_A_loss_real = d_model_A.train_on_batch(img_A, valid)
            d_model_A_loss_fake = d_model_A.train_on_batch(X_fake_A, fake)
            d_model_A_loss = 0.5 * np.add(d_model_A_loss_real, d_model_A_loss_fake)
            # train Discriminator B
            d_model_B_loss_real = d_model_B.train_on_batch(img_B, valid)
            d_model_B_loss_fake = d_model_B.train_on_batch(X_fake_B, fake)
            d_model_B_loss = 0.5 * np.add(d_model_B_loss_real, d_model_B_loss_fake)
            # total loss of Discriminators
            total_d_loss = 0.5 * np.add(d_model_A_loss, d_model_B_loss)
          
            #__________________TRAIN__GENERATORS_____________________
            if opt.increase_generator_iteration:
                for iteration in range(opt.generator_iterations):
                    g_loss = composite_model.train_on_batch([img_A, img_B], [valid, valid, img_A, img_B, img_A, img_B])
            else:
                g_loss = composite_model.train_on_batch([img_A, img_B], [valid, valid, img_A, img_B, img_A, img_B])


            #__________________UPDATING__LEARNING__RATES________________
            if opt.use_lr_decay and epoch > opt.decay_epoch:
                update_learning_rate_2(d_model_A, epoch)
                update_learning_rate_2(d_model_B, epoch)
                update_learning_rate_2(composite_model, epoch)
            
            # Plot the progress
            elapsed_time = datetime.datetime.now() - start_time
            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                    % ( epoch, opt.n_epochs,
                                                                        batch, data_loader.n_batches,
                                                                        total_d_loss[0], 100*total_d_loss[1],
                                                                        g_loss[0],
                                                                        np.mean(g_loss[1:3]),
                                                                        np.mean(g_loss[3:5]),
                                                                        np.mean(g_loss[5:6]),
                                                                        elapsed_time))
            
            # If at save interval => save generated image samples and models
            if batch % opt.sample_interval == 0:
                
                sample_images(epoch, batch)
                g_AtoB.save(f'../models/{opt.dataset_name}/generatorA2B.h5')
                g_BtoA.save(f'../models/{opt.dataset_name}/generatorB2A.h5')
                d_model_A.save(f'../models/{opt.dataset_name}/discriminatorA.h5')
                d_model_B.save(f'../models/{opt.dataset_name}/discriminatorB.h5')
      
