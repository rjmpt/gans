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

import helper


####  VARIATIONAL DIVERGENCE ESTIMATION CLASSES
# https://arxiv.org/abs/1606.00709.pdf
# https://arxiv.org/abs/0809.0853

class FDivTotalVariation():
	def __init__(self, name = '/FDivTotalVariation'):
		self.name = name

	def generator_function(self, u):
		return 0.5*tf.abs(u-1)

	def conjugate_function(self, t):
		return t

	def codomain_function(self, v):
		return 0.5*tf.nn.tanh(v)

class FDivKL():
	def __init__(self, name = '/FDivKL'):
		self.name = name

	def generator_function(self, u):
		return u*helper.tf_safe_log(u)

	def conjugate_function(self, t):
		return tf.exp(t-1)

	def codomain_function(self, v):
		return v

class FDivReverseKL():
	def __init__(self, name = '/FDivReverseKL'):
		self.name = name

	def generator_function(self, u):
		return -helper.tf_safe_log(u)

	def conjugate_function(self, t):
		return -1-helper.tf_safe_log(-t)

	def codomain_function(self, v):
		return -tf.exp(-v)

class FDivPearsonChiSquared():
	def __init__(self, name = '/FDivPearsonChiSquared'):
		self.name = name
	
	def generator_function(self, u):
		return (u-1)**2

	def conjugate_function(self, t):
		return 0.25*(t**2)+t

	def codomain_function(self, v):
		return v

class FDivNeymanChiSquared():
	def __init__(self, name = '/FDivNeymanChiSquared'):
		self.name = name
	
	def generator_function(self, u):
		return ((1-u)**2)/(u+1e-7)

	def conjugate_function(self, t):
		return 2-2*tf.sqrt(1-t)

	def codomain_function(self, v):
		return 1-tf.exp(-v)

class FDivSquaredHellinger():
	def __init__(self, name = '/FDivSquaredHellinger'):
		self.name = name
	
	def generator_function(self, u):
		return (tf.sqrt(u)-1)**2

	def conjugate_function(self, t):
		return t/(1-t)

	def codomain_function(self, v):
		return 1-tf.exp(-v)

class FDivJS():
	def __init__(self, name = '/FDivJS'):
		self.name = name
	
	def generator_function(self, u):
		return -(u+1)*helper.tf_safe_log((u+1)/2)+u*helper.tf_safe_log(u)

	def conjugate_function(self, t):
		return -helper.tf_safe_log(2-tf.exp(t))

	def codomain_function(self, v):
		return np.log(2)-helper.tf_safe_log(1+tf.exp(-v))

class FDivAdHocGANJS():
	def __init__(self, name = '/FDivAdHocGANJS'):
		self.name = name
	
	def generator_function(self, u):
		return -(u+1)*helper.tf_safe_log((u+1))+u*helper.tf_safe_log(u)

	def conjugate_function(self, t):
		return -helper.tf_safe_log(1-tf.exp(t))

	def codomain_function(self, v):
		return -helper.tf_safe_log(1+tf.exp(-v))

class KLDivDiagGaussianVsDiagGaussian():
	def __init__(self, name = 'KLDivDiagGaussianVsDiagGaussian'):
		self.name = name

	def forward(self, dist1, dist2):
		assert (type(dist1) == DiagonalGaussianDistribution)
		assert (type(dist2) == DiagonalGaussianDistribution)
		mean_diff_sq = (dist2.mean - dist1.mean)**2
		log_var1 = 2*dist1.log_std
		log_var2 = 2*dist2.log_std
		scale_factor = tf.exp(-log_var2)
		first = 2*tf.reduce_sum(dist2.log_std-dist1.log_std, axis=1, keepdims=True)
		sec_third = tf.reduce_sum(tf.exp(log_var1-log_var2), axis=1, keepdims=True)-log_var1.get_shape().as_list()[1]
		fourth = tf.reduce_sum((mean_diff_sq*scale_factor), axis=1, keepdims=True)
		KLD_sample = 0.5*(first+sec_third+fourth)
		return KLD_sample

class KLDivDiagGaussianVsNormal():
	def __init__(self, name = 'KLDivDiagGaussianVsNormal'):
		self.name = name

	def forward(self, dist):
		assert (type(dist) == DiagonalGaussianDistribution)
		log_var = dist.log_std*2
		KLD_element = -0.5*((log_var+1)-((dist.mean**2)+(tf.exp(log_var))))
		KLD_sample = tf.reduce_sum(KLD_element, axis=1, keepdims=True)
		return KLD_sample 
