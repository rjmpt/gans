from __future__ import print_function
from IPython.core.debugger import Pdb
pdb = Pdb()
trace = pdb.set_trace

import numpy as np 
np.set_printoptions(suppress=True)
import tensorflow as tf
if tf.__version__ == '1.14.0':
    import tensorflow_probability as tfp
    tf_compat_v1 = tf.compat.v1
    tf_compat_v1.enable_resource_variables()
else:
    tf_compat_v1 = tf

import DNN
import helper

class Model():
    #model initialization (discriminator and generator network setup, visualization setup)
    def __init__(self, global_args=None, config=None, name='VanillaGAN'):
        self.global_args = global_args
        self.config = config
        self.name = name
        self.computations = {}
        assert (self.config.mode == 'Regular' or self.config.mode == 'Alternate')
        if self.global_args.dataset_name == 'CIFAR10': 
            image_size, n_image_channels = [32, 32], 3
     
        self.generator_dnn = DNN.DNN(proto_structure_list = self.config.architecture_function(type='Generator', image_size=image_size, n_out_channels=n_image_channels), name='Generator')
        self.generator_dnn.get_network_activation_sizes([1, 1], verbose=3)
        assert (self.generator_dnn.output_size_for_input_size([1, 1]) == image_size)

        self.critic_dnn = DNN.DNN(proto_structure_list = self.config.architecture_function(type='Critic', image_size=image_size, n_out_channels=1), name='Critic')
        self.critic_dnn.get_network_activation_sizes(image_size, verbose=3)
        assert (self.critic_dnn.output_size_for_input_size(image_size) == [1, 1])

        self.latent_prior_dist_class = self.config.latent_prior_dist_class #set to gaussian

        self.visualizations = {
                               'Training Images': {'variables': ['input_images'], 'pipeline': 'Image Matrix', 'args': {'data_mode': 'Training', 'sample_spec': 64}},
                               'Test Images': {'variables': ['input_images'], 'pipeline': 'Image Matrix', 'args': {'data_mode': 'Test', 'sample_spec': 64}},
                               'Generative Model Sample': {'variables': ['transformed_sample'], 'pipeline': 'Image Matrix', 'args': {'data_mode': 'Test', 'sample_spec': 64}},
                               'Inception Score Generated Samples': {'variables': ['inception_score'], 'pipeline': 'Line Plot', 'args': {'data_mode': 'Test', 'sample_spec': 'All', 'x_axis': 'data_epoch'}},
                               'Critique of Generated Samples': {'variables': ['critique_transformed_sample'], 'pipeline': 'Line Plot', 'args': {'data_mode': 'Test', 'sample_spec': 'All', 'x_axis': 'data_epoch'}}, #['data_epoch', 'epoch', 'hyperparameter_epoch']
                               'Critique of Real Samples': {'variables': ['critique_real_sample'], 'pipeline': 'Line Plot', 'args': {'data_mode': 'Test', 'sample_spec': 'All', 'x_axis': 'data_epoch'}}, #['data_epoch', 'epoch', 'hyperparameter_epoch']
                               'Generator Objective': {'variables': ['generator_objective'], 'pipeline': 'Line Plot', 'args': {'data_mode': 'Test', 'sample_spec': 'All', 'x_axis': 'data_epoch'}},
                               'Critic Objective': {'variables': ['critic_objective'], 'pipeline': 'Line Plot', 'args': {'data_mode': 'Test', 'sample_spec': 'All', 'x_axis': 'data_epoch'}}
                              }
        
    def get_computation(self, device_name):
        if device_name not in self.computations:
            computation = helper.AttrDict(device_name=device_name)
            self.computations[device_name] = computation
        else: print('Model computations for this device are already executed. Aborting.'); quit()
        return computation
    
    #this is where the feedforward inference actually occurs
    def inference(self, input_dict, device_name=None):
        computation = self.get_computation(device_name)
        computation.input_images = input_dict['Image']
        computation.batch_size_tf = tf.shape(computation.input_images)[0]
        
        #use the appropriate latent prior as the experiment args specify
        if self.latent_prior_dist_class.__name__ == 'DiagonalGaussianDistribution':
            computation.latent_prior_dist_params = tf.concat([tf.zeros([computation.batch_size_tf, self.config.latent_dim], tf.float32), 
                                                              tf.zeros([computation.batch_size_tf, self.config.latent_dim], tf.float32)], axis=1) 
        elif self.latent_prior_dist_class.__name__  == 'UniformDistribution':
            computation.latent_prior_dist_params = tf.concat([tf.zeros([computation.batch_size_tf, self.config.latent_dim], tf.float32), 
                                                              tf.ones([computation.batch_size_tf, self.config.latent_dim], tf.float32)], axis=1) 
        elif self.latent_prior_dist_class.__name__ == 'UniformSphereDistribution':
            computation.latent_prior_dist_params = tf.concat([tf.zeros([computation.batch_size_tf, self.config.latent_dim], tf.float32), 
                                                              tf.ones([computation.batch_size_tf, 1], tf.float32)], axis=1)

 
        computation.latent_prior_dist = self.latent_prior_dist_class(params=computation.latent_prior_dist_params, shape=[computation.batch_size_tf, self.config['latent_dim']]) 
        computation.latent_prior_sample = computation.latent_prior_dist.sample()
        
        #get the generated sample and interpolate it with a real image
        computation.transformed_sample_training = self.generator_dnn.forward(computation.latent_prior_sample[:, np.newaxis, np.newaxis, :])
        computation.alpha = tf.random.uniform(shape=[computation.batch_size_tf, 1, 1, 1], minval=0., maxval=1.)
        computation.interpolated_sample = computation.alpha * computation.input_images + (1-computation.alpha) * computation.transformed_sample_training
        
        self.generator_dnn.set_stage('Test')
        computation.transformed_sample = self.generator_dnn.forward(computation.latent_prior_sample[:, np.newaxis, np.newaxis, :]) #used for visualizations

        #run the generated image and real image through discriminator
        computation.critique_transformed_sample_training = self.critic_dnn.forward(computation.transformed_sample_training)[:, 0, 0, :]
        computation.critique_transformed_sample = self.critic_dnn.forward(computation.transformed_sample)[:, 0, 0, :] #used for visualizations
        computation.critique_real_sample = self.critic_dnn.forward(computation.input_images)[:, 0, 0, :] 

        #vanilla GAN loss without the log terms
        if self.config.mode == 'Regular':
            computation.generator_objective = -tf.reduce_mean(computation.critique_transformed_sample_training)
            computation.critic_objective = -tf.reduce_mean(computation.critique_real_sample) - computation.generator_objective 
        elif self.config.mode == 'Alternate': # "non-saturating" generator objective
            computation.generator_objective = -tf.reduce_mean(computation.critique_transformed_sample_training)
            computation.critic_objective = -tf.reduce_mean(computation.critique_real_sample) + tf.reduce_mean(computation.critique_transformed_sample_training)

        #add gradient penalty to critic loss using the hyperparameter lambda_gp
        computation.gradients = tf.gradients(self.critic_dnn.forward(computation.interpolated_sample)[:, 0, 0, :], [computation.interpolated_sample])[0]
        computation.gradient_penalty = tf.reduce_mean((tf.norm(computation.gradients, ord='euclidean')-1.)**2)
        computation.critic_objective = computation.critic_objective + self.config.lambda_gp * computation.gradient_penalty

        computation.objective_dict = {
                                      'Generator': computation.generator_objective, 
                                      'Critic': computation.critic_objective,
                                     }
