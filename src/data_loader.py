from glob import glob
import numpy as np
import cv2
from matplotlib.pyplot import imread
import tensorflow as tf
class DataLoader():
    def __init__(self, dataset, image_shape, batch_size):
        self.dataset_name = dataset
        self.image_shape = image_shape
        self.batch_size = batch_size

    def read_image(self, image_path):
        'Read an image from a file as an array'
        image = imread(image_path, 'RGB').astype(np.float)
        return image

    def load_image(self, image_path):
        'Read and prepare image'
        image = self.read_image(image_path)
        image = cv2.resize(image, self.image_shape)
        image = image/127.5 - 1.
        return image

    def load_dataset(self, domain, batch_size=8, test=False):
        'Load data from dataset folders, and prepare image for batch'
        # Select the folder:  'testA' / 'trainB'...
        image_shape = self.image_shape
        if test:
            batch_size = batch_size
            data_split = f'test{domain}'
        else:
            batch_size = self.batch_size
            data_split = f'train{domain}'

        path = glob(f'../cyclegan_datasets/{self.dataset_name}/{data_split}/*')    

        # define batch
        batch = np.random.choice(a=path, size=batch_size)

        # prepare images
        images = []
        for image in batch:
            # read
            image = self.read_image(image)
            # resize
            if test:
                image = cv2.resize(image, image_shape)
            else:
                image = cv2.resize(image, image_shape)
                # flip
                if np.random.random() > 0.5:
                    image = np.fliplr(img)
            
            images.append(image)
        # normalization
        images = np.array(images)/127.5 - 1.
        
        return images

    def load_batch(self, batch_size=8):
        # define path
        path_A = glob('../cyclegan_datasets/%s/trainA/*' % (self.dataset_name))
        path_B = glob('../cyclegan_datasets/%s/trainB/*' % (self.dataset_name))

        # how many batches are in the dataset
        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size


        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.read_image(img_A)
                img_B = self.read_image(img_B)


                img_A = cv2.resize(img_A, self.image_shape)
                img_B = cv2.resize(img_B, self.image_shape)


                if len(img_A.shape) > 2 and img_A.shape[2] == 4:
                    img_A = img_A[:, :, :3]
                if len(img_B.shape) > 2 and img_B.shape[2] == 4:
                    img_B = img_B[:, :, :3]

                
                if np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)
            
                img_A = tf.keras.preprocessing.image.random_shift(img_A, 0.2, 0.2)
                img_B = tf.keras.preprocessing.image.random_shift(img_B, 0.2, 0.2)
            
                #img_A = tf.keras.preprocessing.image.random_rotation(img_A, 20) --bad
                #img_B = tf.keras.preprocessing.image.random_rotation(img_B, 20) -- bad
            
                #img_A = tf.keras.preprocessing.image.random_zoom(img_A, (-0.2, -0.3)) --black and white
                #img_B = tf.keras.preprocessing.image.random_zoom(img_B, (-0.2, -0.3)) --black and white

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B
    
    def check_min_count(self):
        trainA = glob(f'../cyclegan_datasets/{self.dataset_name}/trainA/*')
        trainB = glob(f'/../cyclegan_datasets/{self.dataset_name}/trainB/*')
        return min(len(trainA), len(trainB))

    def generate_real_samples(self, domain : str, n_samples):
        ''' Select a batch of random samples

        Keyword arguments:
        domain -- domain of images - A / B
        n_samples -- sample size

        Returns:
        X - randomly selected images
        y - PatchGan class labels (1)
        '''

        patch = int(self.image_shape[0] / 2**4)
        data_split = f'train{domain}'
        path = glob(f'../cyclegan_datasets/{self.dataset_name}/{data_split}/*')

        index = np.random.randint(0, len(path), n_samples)
        X = np.array(path)[index]
        y = np.ones((n_samples, n_samples, patch, 1))
        return X, y

    def generate_fake_samples(self, generator_model, dataset):
        ''' Generate a batch of images, returns images and target'''
        patch = int(self.image_shape[0] / 2**4)
        
        X = generator_model.predict(dataset)
        y = np.zeros((len(X), patch, patch, 1))
        return X, y

    def update_image_pool(self, pool, images, pool_size=50):
        '''Filling the image pool to update discriminators,
        returns a list of images selected from the pool'''
        selected = []
        for image in images:
            # fill the pool
            if len(pool) < pool_size:
                pool.append(image)
                selected.append(image)
            # use the image without adding it to the pool
            elif np.random.random() > 0.5:
                selected.append(image)
            else:
                ix = np.random.randint(0, len(pool))
                selected.append(pool[ix])
                pool[ix] = image
        return np.asarray(selected)