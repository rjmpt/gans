import helper 
import distributions
import statistical_divergences
import tensorflow as tf
if tf.__version__ == '1.14.0':
    tf_compat_v1 = tf.compat.v1
    tf_compat_v1.enable_resource_variables()
else:
    tf_compat_v1 = tf

def get_algorithm(algorithm_name):
    if algorithm_name == 'VanillaGAN-DCGAN':
        ## DCGAN
        from architectures import get_dcgan_architectures
        from distributions import DiagonalGaussianDistribution

        algorithm_args = helper.AttrDict({'architecture_function': get_dcgan_architectures, 'latent_prior_dist_class': DiagonalGaussianDistribution, 
                                          'latent_dim': 100, 'mode': 'Alternate', 'batch_size': 32, 'group_freq_dict': {'Generator': 1, 'Critic': 1}})
        optimization_args = {'Generator': helper.AttrDict({'optimizer_class': 'Adam', 'learning_rate': 0.0001, 'learning_rate_decay_rate': 0.0001, 
                                                           'beta1': 0.5, 'beta2': 0.99, 'epsilon': 1e-07, 'gradient_clipping': 0}),
                             'Critic': helper.AttrDict({'optimizer_class': 'Adam', 'learning_rate': 0.0001, 'learning_rate_decay_rate': 0.0001, 
                                                        'beta1': 0.5, 'beta2': 0.99, 'epsilon': 1e-07, 'gradient_clipping': 0}),}
        
        return algorithm_args, optimization_args


    elif algorithm_name == 'fGAN-KL-DCGAN':
        ## DCGAN
        from architectures import get_dcgan_architectures
        from distributions import DiagonalGaussianDistribution
        from statistical_divergences import FDivKL

        algorithm_args = helper.AttrDict({'architecture_function': get_dcgan_architectures, 'latent_prior_dist_class': DiagonalGaussianDistribution,
                                          'f_divergence_class': FDivKL, 'latent_dim': 100, 'mode': 'Alternate', 'batch_size': 32, 
                                          'group_freq_dict': {'Generator': 1, 'Critic': 1}})
        optimization_args = {'Generator': helper.AttrDict({'optimizer_class': 'Adam', 'learning_rate': 0.0001, 'learning_rate_decay_rate': 0.0001, 
                                                           'beta1': 0.5, 'beta2': 0.99, 'epsilon': 1e-07, 'gradient_clipping': 0}),
                             'Critic': helper.AttrDict({'optimizer_class': 'Adam', 'learning_rate': 0.0001, 'learning_rate_decay_rate': 0.0001, 
                                                        'beta1': 0.5, 'beta2': 0.99, 'epsilon': 1e-07, 'gradient_clipping': 0}),}
        
        return algorithm_args, optimization_args


    elif algorithm_name == 'fGAN-reverseKL-DCGAN':
        ## DCGAN
        from architectures import get_dcgan_architectures
        from distributions import DiagonalGaussianDistribution
        from statistical_divergences import FDivReverseKL

        algorithm_args = helper.AttrDict({'architecture_function': get_dcgan_architectures, 'latent_prior_dist_class': DiagonalGaussianDistribution,
                                          'f_divergence_class': FDivReverseKL, 'latent_dim': 100, 'mode': 'Alternate', 'batch_size': 32, 
                                          'group_freq_dict': {'Generator': 1, 'Critic': 1}})
        optimization_args = {'Generator': helper.AttrDict({'optimizer_class': 'Adam', 'learning_rate': 0.0001, 'learning_rate_decay_rate': 0.0001, 
                                                           'beta1': 0.5, 'beta2': 0.99, 'epsilon': 1e-07, 'gradient_clipping': 0}),
                             'Critic': helper.AttrDict({'optimizer_class': 'Adam', 'learning_rate': 0.0001, 'learning_rate_decay_rate': 0.0001, 
                                                        'beta1': 0.5, 'beta2': 0.99, 'epsilon': 1e-07, 'gradient_clipping': 0}),}
        
        return algorithm_args, optimization_args


    elif algorithm_name == 'fGAN-Pearson-DCGAN':
        ## DCGAN
        from architectures import get_dcgan_architectures
        from distributions import DiagonalGaussianDistribution
        from statistical_divergences import FDivPearsonChiSquared

        algorithm_args = helper.AttrDict({'architecture_function': get_dcgan_architectures, 'latent_prior_dist_class': DiagonalGaussianDistribution,
                                          'f_divergence_class': FDivPearsonChiSquared, 'latent_dim': 100, 'mode': 'Alternate', 'batch_size': 32, 
                                          'group_freq_dict': {'Generator': 1, 'Critic': 1}})
        optimization_args = {'Generator': helper.AttrDict({'optimizer_class': 'Adam', 'learning_rate': 0.0001, 'learning_rate_decay_rate': 0.0001, 
                                                           'beta1': 0.5, 'beta2': 0.99, 'epsilon': 1e-07, 'gradient_clipping': 0}),
                             'Critic': helper.AttrDict({'optimizer_class': 'Adam', 'learning_rate': 0.0001, 'learning_rate_decay_rate': 0.0001, 
                                                        'beta1': 0.5, 'beta2': 0.99, 'epsilon': 1e-07, 'gradient_clipping': 0}),}
        
        return algorithm_args, optimization_args


    elif algorithm_name == 'wGAN_gp-DCGAN':
        ## DCGAN
        from architectures import get_dcgan_architectures_wgan
        from distributions import DiagonalGaussianDistribution

        algorithm_args = helper.AttrDict({'architecture_function': get_dcgan_architectures_wgan, 'latent_prior_dist_class': DiagonalGaussianDistribution, 
                                          'latent_dim': 100, 'lambda_gp': 10, 'mode': 'Alternate', 'batch_size': 32, 'group_freq_dict': {'Generator': 1, 'Critic': 5}})
        optimization_args = {'Generator': helper.AttrDict({'optimizer_class': 'Adam', 'learning_rate': 0.0001, 'learning_rate_decay_rate': 0.0001, 
                                                           'beta1': 0, 'beta2': 0.9, 'epsilon': 1e-07, 'gradient_clipping': 0}),
                             'Critic': helper.AttrDict({'optimizer_class': 'Adam', 'learning_rate': 0.0001, 'learning_rate_decay_rate': 0.0001, 
                                                        'beta1': 0, 'beta2': 0.9, 'epsilon': 1e-07, 'gradient_clipping': 0}),}
        
        return algorithm_args, optimization_args
