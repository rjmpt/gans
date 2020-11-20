#this is a utility class which provides sampling functionality as well as log-pdf evaluation for numerous
#n-dimensional prior distributions, which is very useful when wishing to experiment with various priors

from __future__ import print_function
from IPython.core.debugger import Pdb
pdb = Pdb()
trace = pdb.set_trace

import math
import scipy as scipy

import numpy as np 
np.set_printoptions(suppress=True)
import tensorflow as tf
if tf.__version__ == '1.14.0':
    import tensorflow_probability as tfp
    tf_compat_v1 = tf.compat.v1
    tf_compat_v1.enable_resource_variables()
else:
    tf_compat_v1 = tf

import helper

####  UNIFORM DISTRIBUTION
class UniformDistribution():
	def __init__(self, params=None, shape=None, name='UniformDistribution'):
		if len(params.get_shape().as_list()) == 2:
			self.low = params[:, :int(params.get_shape().as_list()[1]/2.)]
			self.high = params[:, int(params.get_shape().as_list()[1]/2.):]
			
		self.name = name

	def num_params(num_dim):
		return 2*num_dim

	def get_interpretable_params(self):
		return [self.low, self.high]

	def sample(self, b_mode=False):
		if b_mode: sample = (self.low+self.high)/2.
		else: sample = tf.random_uniform(tf.shape(self.low), 0, 1, dtype=tf.float32)*(self.high-self.low)+self.low
		return sample

	def log_pdf(self, sample):
		assert (len(sample.get_shape())==2)
		log_volume = tf.reduce_sum(helper.tf_safe_log(self.high-self.low), axis=1, keepdims=True)
		return 0-log_volume


####  UNIFORM n-SPHERE DISTRIBUTION
class UniformSphereDistribution():
	def __init__(self, params=None, shape=None, name='UniformSphereDistribution'):
		if len(params.get_shape().as_list()) == 2: 
			self.centers = params[:, :-1]
			self.radius = params[:, -1, np.newaxis]		
		self.name = name
		self.dim = self.centers.get_shape().as_list()[1]

	def num_params(num_dim):
		return num_dim+1

	def get_interpretable_params(self):
		return [self.centers, self.radius]

	def sample(self):
		dir_normal = tf.random.normal(shape=tf.shape(self.centers))
		dir_normal_norm = helper.safe_tf_sqrt(tf.reduce_sum(dir_normal**2, axis=1, keepdims=True))
		sample_dir = dir_normal/dir_normal_norm
		sample = self.radius*sample_dir+self.centers
		return sample

	def log_pdf(self, sample):
		pdb.set_trace()
		assert (len(sample.get_shape())==2)
		log_volume = (self.dim/2)*np.log(np.pi)-(scipy.special.gammaln(1+self.dim/2))+self.dim*helper.tf_safe_log(self.radius)
		return 0-log_volume

	
####  UNIFORM n-BALL DISTRIBUTION
class UniformBallDistribution():
	def __init__(self, params=None, shape=None, name='UniformBallDistribution'):
		if len(params.get_shape().as_list()) == 2: 
			self.centers = params[:, :-1]
			self.radius = params[:, -1, np.newaxis]		
		self.name = name
		self.dim = self.centers.get_shape().as_list()[1]

	def num_params(num_dim):
		return num_dim+1

	def get_interpretable_params(self):
		return [self.centers, self.radius]

	def sample(self):
		dir_normal = tf.random.normal(shape=tf.shape(self.centers))
		dir_normal_norm = helper.safe_tf_sqrt(tf.reduce_sum(dir_normal**2, axis=1, keepdims=True))
		sample_dir = dir_normal/dir_normal_norm
		sample_norm = tf.random_uniform(tf.shape(self.radius), 0, 1, dtype=tf.float32)*self.radius
		sample = sample_norm*sample_dir+self.centers
		return sample

	def log_pdf(self, sample):
		assert (len(sample.get_shape())==2)
		log_volume = (self.dim/2)*np.log(np.pi)-(scipy.special.gammaln(1+self.dim/2))+self.dim*helper.tf_safe_log(self.radius)
		return 0-log_volume


####  DIAGONAL GAUSSIAN DISTRIBUTION
class DiagonalGaussianDistribution():
	def __init__(self, params=None, shape = None, mode='exponential', bound_range=[1e-7, 2.], name = 'DiagonalGaussianDistribution'):
		self.mode = mode
		self.bound_range = bound_range
		assert (self.mode == 'exponential' or self.mode == 'softplus' or self.mode == 'bounded')
		assert (self.bound_range[0] > 0 and self.bound_range[1] > 0)

		if len(params.get_shape().as_list()) == 2: 
			self.mean = params[:, :int(params.get_shape().as_list()[1]/2.)]
			self.pre_std = params[:, int(params.get_shape().as_list()[1]/2.):]

			if self.mode == 'bounded':
				self.std = self.bound_range[0]+tf.nn.sigmoid(self.pre_std)*(self.bound_range[1]-self.bound_range[0])
				self.var = self.std**2
				self.log_std = helper.tf_safe_log(self.std)
			elif self.mode == 'exponential':
				self.log_std = self.pre_std
				self.std = tf.exp(self.log_std)
				self.var = tf.exp(2*self.log_std)
			elif self.mode == 'softplus':
				self.std = 1e-5+tf.nn.softplus(self.pre_std)
				self.var = self.std**2
				self.log_std = helper.tf_safe_log(self.std)

		else: print('Invalid Option. DiagonalGaussianDistribution.'); quit()
		self.name = name

	@staticmethod
	def num_params(num_dim):
		return 2*num_dim
		
	def get_interpretable_params(self):
		return [self.mean, self.std]

	def sample(self, b_mode=False):
		if b_mode: 
			sample = self.mean
		else:
			eps = tf.random.normal(shape=tf.shape(self.log_std))
			sample = (self.std*eps)+self.mean
		return sample

	def log_pdf(self, sample):
		assert (len(sample.get_shape())==2)		
		unnormalized_log_prob = -0.5*tf.reduce_sum(((self.mean-sample)**2)/(1e-7+self.var), axis=1, keepdims=True)
		log_partition = -0.5*tf.reduce_sum(2*self.log_std, axis=1, keepdims=True)+math.log(2*math.pi)*(-self.mean.get_shape().as_list()[1]/2.0)
		log_prob = unnormalized_log_prob+log_partition
		return log_prob
