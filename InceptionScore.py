# Reference sources:
##   https://github.com/taki0112/GAN_Metrics-Tensorflow/blob/master/main.py
##   tensorflow/tensorflow/models/image/imagenet/classify_image.py

import numpy as np 
np.set_printoptions(suppress=True)
import tensorflow as tf
if tf.__version__ == '1.14.0':
    import tensorflow_probability as tfp
    tf_compat_v1 = tf.compat.v1
    tf_compat_v1.enable_resource_variables()
else:
    tf_compat_v1 = tf

import functools
import numpy as np
import time
from tensorflow.python.ops import array_ops
from scipy import misc

tfgan = tf.contrib.gan
session = tf.InteractiveSession()

def inception_logits(images, num_splits = 1):
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.image.resize_bilinear(images, [size, size])
    generated_images_list = tf.split(images, num_or_size_splits = num_splits)
    logits = tf.map_fn(
        fn = functools.partial(tfgan.eval.run_inception, output_tensor = 'logits:0'),
        elems = tf.stack(generated_images_list),
        parallel_iterations = 1,
        back_prop = False,
        swap_memory = True,
        name = 'RunClassifier')
    logits = tf.concat(tf.unstack(logits), 0)
    return logits

def get_inception_probs(batch_size, images, inception_images, logits):
    n_batches = len(images) // batch_size
    preds = np.zeros([n_batches * batch_size, 1000], dtype = np.float32)
    for i in range(n_batches):
        inp = images[i * batch_size:(i + 1) * batch_size] #/ 255. * 2 - 1
        preds[i * batch_size:(i + 1) * batch_size] = logits.eval({inception_images:inp})[:, :1000]
    preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
    return preds

def preds2score(preds, splits=10):
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def get_inception_score(batch_size, images, inception_images, logits, splits=10):
    assert(type(images) == np.ndarray)
    assert(len(images.shape) == 4)
    assert(images.shape[1] == 3)
    #assert(np.min(images[0]) >= 0 and np.max(images[0]) > 10), 'Image values should be in the range [0, 255]'
    #print('Calculating Inception Score with %i images in %i splits' % (images.shape[0], splits))
    start_time=time.time()
    preds = get_inception_probs(batch_size, images, inception_images, logits)
    mean, std = preds2score(preds, splits)
    #print('Inception Score calculation time: %f s' % (time.time() - start_time))
    return mean, std  # Reference values: 11.34 for 49984 CIFAR-10 training set images, or mean=11.31, std=0.08 if in 10 splits.

def inception_score(images):
    images = np.transpose(images, axes=[0, 3, 1, 2])

    # A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
    BATCH_SIZE = 64

    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])

    logits = inception_logits(inception_images)

    return get_inception_score(BATCH_SIZE, images, inception_images, logits, splits=10)

'''
from DataLoaders.CIFAR.Cifar10Loader import DataLoader
data_loader = DataLoader(batch_size=6400)

for i, curr_batch_size, batch_np in data_loader:
    inception_score(batch_np['Image'])
'''
