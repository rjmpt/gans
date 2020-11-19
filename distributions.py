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

class UniformDiscreteDistribution():
	def __init__(self, params=None, interval=[1, 0, 9], shape=None, name='UniformDiscreteDistribution'):
		if params is not None: 
			assert (len(params.get_shape().as_list()) == 2) 
			self.list_values = params
		else:
			self.list_values = interval
		self.name = name
		 # list_values = np.linspace(1,10,10).astype(int)

	def num_params(num_dim):
		return 0

	def get_interpretable_params(self):
		return self.list_values

	def sample(self, b_mode=False):
		if isinstance(self.list_values, (list,)):
			float_sample = (tf.random_uniform((self.list_values[0], 1), 0, 1, dtype=tf.float32)*((self.list_values[2]+1)-self.list_values[1])+self.list_values[1])
			int_sample = tf.cast(tf.cast(tf.floor(float_sample), 'int32'), 'float32')
			sample = tf.stop_gradient(int_sample)
		else:
			if b_mode: sample = (self.low+self.high)/2.
			else: sample = tf.random_uniform(tf.shape(self.low), 0, 1, dtype=tf.float32)*(self.high-self.low)+self.low
		return sample

	def log_pdf(self, sample):
		assert (len(sample.get_shape())==2)
		log_volume = tf.reduce_sum(helper.tf_safe_log(self.high-self.low), axis=1, keepdims=True)
		return 0-log_volume

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

####  UNIFORM Hallow n-BALL DISTRIBUTION

class UniformHollowBallDistribution():
	def __init__(self, params=None, shape=None, name='UniformHollowBallDistribution'):
		if len(params.get_shape().as_list()) == 2: 
			self.centers = params[:, :-2]
			self.inner_radius = params[:, -2, np.newaxis]		
			self.outer_radius = params[:, -1, np.newaxis]		
		self.name = name
		self.dim = self.centers.get_shape().as_list()[1]

	def num_params(num_dim):
		return num_dim+2

	def get_interpretable_params(self):
		return [self.centers, self.inner_radius, self.outer_radius]

	def sample(self):
		dir_normal = tf.random.normal(shape=tf.shape(self.centers))
		dir_normal_norm = helper.safe_tf_sqrt(tf.reduce_sum(dir_normal**2, axis=1, keepdims=True))
		sample_dir = dir_normal/dir_normal_norm
		sample_norm = tf.random_uniform(tf.shape(self.inner_radius), 0, 1, dtype=tf.float32)*(self.outer_radius-self.inner_radius)+self.inner_radius
		sample = sample_norm*sample_dir+self.centers
		return sample

	def log_pdf(self, sample):
		assert (len(sample.get_shape())==2)
		log_inner_volume = (self.dim/2)*np.log(np.pi)-(scipy.special.gammaln(1+self.dim/2))+self.dim*helper.tf_safe_log(self.inner_radius)
		log_outer_volume = (self.dim/2)*np.log(np.pi)-(scipy.special.gammaln(1+self.dim/2))+self.dim*helper.tf_safe_log(self.outer_radius)
		# log_hollow_volume = log_inner_volume+helper.tf_safe_log(tf.exp(log_outer_volume-log_inner_volume)-1)
		log_hollow_volume = log_outer_volume+helper.tf_safe_log(1-tf.exp(log_inner_volume-log_outer_volume))
		return 0-log_hollow_volume

####  BERNOULLI DISTRIBUTION

class BernoulliDistribution():
	def __init__(self, params=None, shape=None, b_sigmoid=True, name='BernoulliDistribution'):
		if len(params.get_shape().as_list()) == 2: 
			if b_sigmoid:
				self.mean = tf.nn.sigmoid(params)
			else:
				self.mean = params				
		else: print('Invalid Option. BernoulliDistribution.'); quit()
		self.name = name
		# self._dist = tf.distributions.Bernoulli(probs=self.mean)

	def num_params(num_dim):
		return num_dim

	def get_interpretable_params(self):
		return [self.mean]

	def sample(self, b_mode=False):
		if b_mode: sample = self.mean #sample = tf.stop_gradient(tf.cast(self.mean > 0.5, tf.float32))
		else: 
			pdb.set_trace()
			# sample = tf.stop_gradient(tf.cast(tf.random_uniform(shape=(tf.shape(self.mean)[0], 1)) < self.mean, tf.float32))
			# sample = tf.stop_gradient(tf.cast(tf.random_uniform(shape=tf.shape(self.mean)) < self.mean, tf.float32))
		return sample

	def log_pdf(self, sample):
		assert (len(sample.get_shape())==2)
		# sample = tf.reshape(sample, [tf.shape(sample)[0], -1])
		binary_cross_entropy_dims = sample*helper.tf_safe_log(self.mean)+ (1.0-sample)*helper.tf_safe_log(1.0-self.mean)
		log_prob = tf.reduce_sum(binary_cross_entropy_dims, axis=1, keepdims=True)
		
		# log_prob = tf.reduce_sum(self._dist._log_prob(sample), axis=1, keepdims=True)
		return log_prob

####  CATEGORICAL/Boltzman DISTRIBUTION

class BoltzmanDistribution():
	def __init__(self, params=None, temperature=1, shape=None, name='BoltzmanDistribution'):
		if len(params.get_shape().as_list()) == 2: self.logits = params
		else: print('Invalid Option. BoltzmanDistribution.'); quit()
		self.temperature = temperature
		self.name = name

	@staticmethod
	def num_params(num_dim):
		return num_dim

	def get_interpretable_params(self):
		return [tf.nn.softmax(self.logits/self.temperature)]

	def sample(self, b_mode=False):
		prob_w = tf.nn.softmax(self.logits/self.temperature)
		if b_mode: 
			indices = tf.argmax(prob_w, axis=1)
		else:
			sed = tf.random_uniform(shape=(prob_w.get_shape().as_list()[0], 1))
			prob_w_accum = tf.cumsum(prob_w, axis=1)
			indices = tf.argmax(tf.cast(prob_w_accum>sed, tf.float32), axis=1)
		sample = tf.stop_gradient(tf.cast(tf.one_hot(indices, depth=prob_w.get_shape().as_list()[1], on_value=1, off_value=0, axis=1), tf.float32))
		return sample

	def entropy(self):
		log_prob_all = tf.nn.log_softmax(self.logits/self.temperature)
		prob_all = tf.exp(log_prob_all)
		entropy = tf.reduce_sum(-log_prob_all*prob_all, axis=1, keepdims=True)
		return entropy

	def log_pdf(self, sample_one_hot):
		assert (len(sample_one_hot.get_shape())==2)
		log_prob_all = tf.nn.log_softmax(self.logits/self.temperature)
		log_prob = tf.reduce_sum(log_prob_all*sample_one_hot, axis=1, keepdims=True)
		return log_prob
	
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

####  DIAGONAL BETA DISTRIBUTION

class DiagonalBetaDistribution():
	def __init__(self, params=None, shape = None, name = 'DiagonalBetaDistribution'):
		if len(params.get_shape().as_list()) == 2: 
			self.alpha = params[:, :int(params.get_shape().as_list()[1]/2.)]
			self.beta = params[:, int(params.get_shape().as_list()[1]/2.):]
		else: print('Invalid Option. DiagonalBetaDistribution.'); quit()
		self.name = name

	@staticmethod
	def num_params(num_dim):
		return 2*num_dim
		
	def get_interpretable_params(self):
		return [self.alpha, self.beta]

	def sample(self, b_mode=False):
		if b_mode: 
			sample = (self.alpha-1)/(self.alpha+self.beta-2)
		else:
			sample = helper.beta_samples(self.alpha, self.beta)
		return sample

	def log_pdf(self, sample):
		assert (len(sample.get_shape())==2)
		unnormalized_log_prob = tf.reduce_sum((self.alpha-1)*helper.tf_safe_log(tf.nn.relu(sample))+(self.beta-1)*helper.tf_safe_log(tf.nn.relu(1-sample)), axis=1, keepdims=True)
		log_partition = tf.reduce_sum(-helper.log_gamma(self.alpha)-helper.log_gamma(self.beta)+helper.log_gamma(self.alpha+self.beta), axis=1, keepdims=True)
		log_prob = unnormalized_log_prob+log_partition
		return log_prob

####  DIAGONAL KUMARASWAMY DISTRIBUTION

class DiagonalKumaraswamyDistribution():
	def __init__(self, params=None, shape = None, b_softplus=True, name = 'DiagonalKumaraswamyDistribution'):
		if len(params.get_shape().as_list()) == 2: 
			if b_softplus:
				self.alpha = tf.nn.softplus(params[:, :int(params.get_shape().as_list()[1]/2.)])
				self.beta = tf.nn.softplus(params[:, int(params.get_shape().as_list()[1]/2.):])
			else:
				self.alpha = params[:, :int(params.get_shape().as_list()[1]/2.)]
				self.beta = params[:, int(params.get_shape().as_list()[1]/2.):]
		else: print('Invalid Option. DiagonalKumaraswamyDistribution.'); quit()
		self.name = name

	@staticmethod
	def num_params(num_dim):
		return 2*num_dim
		
	def get_interpretable_params(self):
		return [self.alpha, self.beta]

	def sample(self, b_mode=False):
		if b_mode: 
			print('Not implemented, mode sample. DiagonalKumaraswamyDistribution.')
			quit()
		else:
			sed = tf.random_uniform(shape=(tf.shape(self.alpha)[0], 1))
			sample = tf.math.pow((1-tf.math.pow(sed, 1/self.beta)), 1/self.alpha)
		return sample

	def log_pdf(self, sample):
		assert (len(sample.get_shape())==2)
		log_prob =  helper.tf_safe_log(self.alpha)+helper.tf_safe_log(self.beta)+(self.alpha-1)*helper.tf_safe_log(sample)+(self.beta-1)*helper.tf_safe_log(1-tf.math.pow(sample, self.alpha))
		return log_prob

####  DIAGONAL LOGIT-NORMAL DISTRIBUTION
class DiagonalLogitNormalDistribution():
	def __init__(self, params=None, shape = None, unimodal=True, name = 'DiagonalLogitNormalDistribution'):
		if len(params.get_shape().as_list()) == 2: 
			if unimodal:
				self.gaussian_dist = DiagonalGaussianDistribution(params = params, mode = 'bounded', bound_range=[1e-7, np.sqrt(2)])
			else:
				self.gaussian_dist = DiagonalGaussianDistribution(params = params)
		else: print('Invalid Option. DiagonalLogitNormalDistribution.'); quit()
		self.name = name

	@staticmethod
	def num_params(num_dim):
		return 2*num_dim
		
	def get_interpretable_params(self):
		return self.gaussian_dist.get_interpretable_params()

	def sample(self, b_mode=False):
		sample = tf.nn.sigmoid(self.gaussian_dist.sample(b_mode=b_mode))
		return sample

	def log_pdf(self, sample):
		assert (len(sample.get_shape())==2)		
		log_sample = helper.tf_safe_log(sample)
		log_one_min_sample = helper.tf_safe_log(1-sample)
		logit_sample= log_sample - log_one_min_sample

		gaussian_log_pdf = self.gaussian_dist.log_pdf(logit_sample)
		log_prob = gaussian_log_pdf - tf.reduce_sum(log_sample+log_one_min_sample, axis=1)[:,np.newaxis]
		return log_prob

####  MIXTURE DISTRIBUTION

class MixtureDistribution():
	def __init__(self, dists, weights, name = 'MixtureDistribution'):
		if np.abs(np.sum(weights)-1) > 1e-7 or len(dists) != len(weights): print('Invalid Option. MixtureDistribution.'); quit()
		self.dists = dists
		self.weights = weights
		self.name = name
		
	def sample(self, b_mode=False):
		cdf = np.cumsum(self.weights)
		uniform_sample = tf.random_uniform((tf.shape(self.dists[0].mean)[0], 1), 0, 1, dtype=tf.float32)
		mixture_indeces = len(self.weights)-tf.reduce_sum(tf.cast(uniform_sample <= cdf[np.newaxis, :], tf.float32), axis=1, keepdims=True)

		sample = None
		for i in range(len(self.weights)):
			mix_mask = tf.cast((mixture_indeces-i) <= 0, tf.float32) * tf.cast((mixture_indeces-i) >= 0, tf.float32)
			mix_samples = self.dists[i].sample(b_mode=b_mode)
			if sample is None: sample = mix_mask*mix_samples
			else: sample = sample + mix_mask*mix_samples
		return sample
		
	def log_pdf(self, sample):
		assert (len(sample.get_shape())==2)
		log_weighted_log_probs = []
		for i in range(len(self.weights)):
			mix_log_prob = self.dists[i].log_pdf(sample)
			log_weighted_log_probs.append(np.log(self.weights[i])+mix_log_prob)
		log_weighted_log_probs_tf = tf.concat(log_weighted_log_probs, axis=1)
		maxes = tf.reduce_max(log_weighted_log_probs_tf, axis=1, keepdims=True)
		log_prob = maxes+helper.tf_safe_log(tf.reduce_sum(tf.exp(log_weighted_log_probs_tf-maxes), axis=1, keepdims=True))
		return log_prob

####  BERNOULLI DISTRIBUTION

class DiracDistribution():
	def __init__(self, params=None, shape=None, name='DiracDistribution'):
		if len(params.get_shape().as_list()) == 2: self.mean = params
		else: print('Invalid Option. DiracDistribution.'); quit()
		self.name = name

	def num_params(num_dim):
		return num_dim

	def get_interpretable_params(self):
		return [self.mean]

	def sample(self, b_mode=False):
		if b_mode: sample = self.mean
		else: sample = self.mean
		return sample

	def log_pdf(self, sample):
		print('Requested pdf of dirac delta sample.')
		return None

####  PRODUCT DISTRIBUTION

class ProductDistribution():
	def __init__(self, sample_properties = None, params = None, name = 'ProductDistribution'):
		self.name = name
		self.sample_properties = sample_properties

		self.dist_class_list = {'flat': [DistributionsAKA[e['dist']] for e in self.sample_properties['flat']], 
								'image': [DistributionsAKA[e['dist']] for e in self.sample_properties['image']]}

		self.param_sizes = {'flat': [self.dist_class_list['flat'][i].num_params(np.prod(e['size'][2:])) for i, e in enumerate(self.sample_properties['flat'])], 
							'image': [self.dist_class_list['image'][i].num_params(np.prod(e['size'][2:])) for i, e in enumerate(self.sample_properties['image'])]}

		self.params = {'flat': helper.split_tensor_tf(params['flat'], -1, [int(self.param_sizes['flat'][i]/np.prod(e['size'][2:-1])) for i, e in enumerate(self.sample_properties['flat'])]), 
					   'image':helper.split_tensor_tf(params['image'], -1, [int(self.param_sizes['image'][i]/np.prod(e['size'][2:-1])) for i, e in enumerate(self.sample_properties['image'])])} 
		
		self.params_flat = {'flat': [tf.reshape(self.params['flat'][i], [-1, np.prod(self.params['flat'][i].get_shape().as_list()[2:])]) for i, e in enumerate(self.params['flat'])],
					 		'image': [tf.reshape(self.params['image'][i], [-1, np.prod(self.params['image'][i].get_shape().as_list()[2:])]) for i, e in enumerate(self.params['image'])]}

		self.dist_list = {'flat': [dist(params = self.params_flat['flat'][i]) for i, dist in enumerate(self.dist_class_list['flat'])],
						  'image': [dist(params = self.params_flat['image'][i]) for i, dist in enumerate(self.dist_class_list['image'])]}

	def sample(self, b_mode=False):
		samples = {'flat': None, 'image': None}
		for obs_type in ['flat', 'image']:
			samples_curr = []
			for i, dist in enumerate(self.dist_list[obs_type]):
				sample = tf.reshape(dist.sample(b_mode), [-1, *self.sample_properties[obs_type][i]['size'][1:]])
				samples_curr.append(sample)
			if len(samples_curr) > 0: 
				samples[obs_type] = tf.concat(samples_curr, axis=-1)
		return samples

	def entropy(self):
		entropies_all = None
		for obs_type in ['flat', 'image']:
			if self.params[obs_type] is not None:
				entropies = []
				for i, e in enumerate(self.dist_list[obs_type]):
					entropies.append(tf.reshape(e.entropy(), [-1, self.params[obs_type][i].get_shape().as_list()[1], 1]))
				if entropies_all is None: entropies_all = entropies
				else: entropies_all = entropies_all + entropies
		return helper.list_sum(entropies_all)

	def log_pdf(self, sample):
		log_pdfs_all = None
		for obs_type in ['flat', 'image']:
			if sample[obs_type] is not None:
				split = helper.split_tensor_tf(sample[obs_type], -1, [e['size'][-1] for e in self.sample_properties[obs_type]])
				split_batched = [tf.reshape(e, [-1, *e.get_shape().as_list()[2:]]) for e in split]
				log_pdfs = []
				for i, e in enumerate(self.dist_list[obs_type]):
					split_batched_flat = tf.reshape(split_batched[i], [-1, np.prod(split_batched[i].get_shape().as_list()[1:])])
					log_pdfs.append(tf.reshape(e.log_pdf(split_batched_flat), [-1, sample[obs_type].get_shape().as_list()[1], 1]))
				if log_pdfs_all is None: log_pdfs_all = log_pdfs
				else: log_pdfs_all = log_pdfs_all + log_pdfs
		return helper.list_sum(log_pdfs_all)

####  TRANSFORMED DISTRIBUTION

class TransformedDistribution():
	"""
	Args:
		base_dist: base distribution.
		transforms_list: list of transforms
		batch_size : Batch size for distribution functions. 

	Raises:
		ValueError: 
	"""
	def __init__(self, base_dist=None, transforms = [], name='transform_dist'):
		self._base_dist = base_dist
		self._transforms = transforms
		self._log_pdf_cache = {}
		self._base_log_pdf_cache = {}

	@property
	def dim(self):
		return self._transforms[len(self._transforms)-1].output_dim

	def add_transforms(self, transforms_list):
		self._transforms = self._transforms + transforms_list

	def sample(self, base_sample=None, base_log_pdf=None):
		if base_sample is None or base_log_pdf is None:
			if self._base_dist is None:
				print('No base sample or distribution given.'); quit()
			else:
				base_sample = self._base_dist.sample()
				base_log_pdf = self._base_dist.log_pdf(base_sample)
		curr_sample, curr_log_pdf = base_sample, base_log_pdf 
		for i in range(len(self._transforms)): 
			curr_sample, curr_log_pdf = self._transforms[i].transform(curr_sample, curr_log_pdf)
		self._log_pdf_cache[curr_sample] = curr_log_pdf
		self._base_log_pdf_cache[base_sample] = base_log_pdf
		return curr_sample, base_sample

	def log_pdf(self, sample):
		if sample not in self._log_pdf_cache: 
			print('Requested log pdf of sample not generated by distribution.'); quit()
		return self._log_pdf_cache[sample]
	
	def base_log_pdf(self, sample):
		if sample not in self._base_log_pdf_cache: 
			print('Requested log pdf of base sample not generated by distribution.'); quit()
		return self._base_log_pdf_cache[sample]




