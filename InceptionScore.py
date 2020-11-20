# Reference sources:
##   This code was obtained from: https://github.com/taki0112/GAN_Metrics-Tensorflow/blob/master/main.py
##   This code references: tensorflow/tensorflow/models/image/imagenet/classify_image.py

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

#get the inception v3 logits for the provided images
def inception_logits(images, num_splits = 1):
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.image.resize_bilinear(images, [size, size])
    generated_images_list = tf.split(images, num_or_size_splits = num_splits)
    
    #get the logits using tfgan
    logits = tf.map_fn(
        fn = functools.partial(tfgan.eval.run_inception, output_tensor = 'logits:0'),
        elems = tf.stack(generated_images_list),
        parallel_iterations = 1,
        back_prop = False,
        swap_memory = True,
        name = 'RunClassifier')
    logits = tf.concat(tf.unstack(logits), 0)
    return logits

#perform softmax operation on the logits to get predictions
def get_inception_probs(batch_size, images, inception_images, logits):
    n_batches = len(images) // batch_size
    preds = np.zeros([n_batches * batch_size, 1000], dtype = np.float32)
    for i in range(n_batches):
        inp = images[i * batch_size:(i + 1) * batch_size] 
        preds[i * batch_size:(i + 1) * batch_size] = logits.eval({inception_images:inp})[:, :1000] #evaluate logits for the current batch of input images
    preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True) #perform softmax
    return preds

#compute kl divergence between marginal p(y) and conditional p(y|x)
def preds2score(preds, splits=10):
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl)) #add exp(kl divergence)
    return np.mean(scores), np.std(scores)

def get_inception_score(batch_size, images, inception_images, logits, splits=10):
    assert(type(images) == np.ndarray)
    assert(len(images.shape) == 4)
    assert(images.shape[1] == 3) 
    
    preds = get_inception_probs(batch_size, images, inception_images, logits)
    return preds2score(preds, splits) 

#function used to obtain inception score for batch of images (np array, [batch_size * l * w * c])
def inception_score(images):
    images = np.transpose(images, axes=[0, 3, 1, 2])

    # A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
    BATCH_SIZE = 64

    # Run images through Inception.
    inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
    logits = inception_logits(inception_images)
    return get_inception_score(BATCH_SIZE, images, inception_images, logits, splits=10)

