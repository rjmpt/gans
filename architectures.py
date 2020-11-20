#This file is a helper which returns various architecture definitions depending on the input image size

import helper 
import tensorflow as tf
if tf.__version__ == '1.14.0':
    tf_compat_v1 = tf.compat.v1
    tf_compat_v1.enable_resource_variables()
else:
    tf_compat_v1 = tf

#All DCGAN architectures: https://arxiv.org/pdf/1511.06434.pdf
def get_dcgan_architectures(type='Generator', image_size=[28, 28], n_out_channels=1):
    if type == 'Generator' and image_size == [28, 28]:
        return [
                (['input_node'], 'layer_1', 'TransposedConvolution', {'n_out_channels': 1024, 'kernel_shape': [4, 4], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': False, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_1'], 'layer_2', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_2'], 'layer_3', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_3'], 'layer_4', 'TransposedConvolution', {'n_out_channels': 512, 'kernel_shape': [5, 5], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': False, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_4'], 'layer_5', 'Crop', {'crop_size_structure': [(0, 1), (0, 1)]}),
                (['layer_5'], 'layer_6', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_6'], 'layer_7', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_7'], 'layer_8', 'TransposedConvolution', {'n_out_channels': 256, 'kernel_shape': [5, 5], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': False, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_8'], 'layer_9', 'Crop', {'crop_size_structure': [(2, 1), (2, 1)]}),
                (['layer_9'], 'layer_10', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_10'], 'layer_11', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_11'], 'layer_12', 'TransposedConvolution', {'n_out_channels': n_out_channels, 'kernel_shape': [5, 5], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': False, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_12'], 'layer_13', 'Crop', {'crop_size_structure': [(1, 2), (1, 2)]}),
                (['layer_13'], 'layer_14', 'ElementwiseApply', {'func': tf.nn.sigmoid}),
               ]
    if type == 'Critic' and image_size == [28, 28]:
        # Critic is very sensitive to slight shifts in the inputs, so we added a ConstantShiftScale class to DNN to 
        # handle differences between the Generator Output range and Critic input range
        return [
                (['input_node'], 'layer_1', 'ConstantScaleShift', {'scale': 2., 'shift': -1.}),
                (['layer_1'], 'layer_2', 'Convolution', {'n_out_channels': 128, 'kernel_shape': [3, 3], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_2'], 'layer_3', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_3'], 'layer_4', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_4'], 'layer_5', 'Convolution', {'n_out_channels': 256, 'kernel_shape': [3, 3], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_5'], 'layer_6', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_6'], 'layer_7', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_7'], 'layer_8', 'Convolution', {'n_out_channels': 512, 'kernel_shape': [3, 3], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_8'], 'layer_9', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_9'], 'layer_10', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_10'], 'layer_11', 'Convolution', {'n_out_channels': 1024, 'kernel_shape': [3, 3], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_11'], 'layer_12', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_12'], 'layer_13', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_13'], 'layer_14', 'Convolution', {'n_out_channels': 1, 'kernel_shape': [2, 2], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
               ]
    if type == 'Generator' and image_size == [32, 32]:
        return [   
                (['input_node'], 'layer_1', 'TransposedConvolution', {'n_out_channels': 1024, 'kernel_shape': [4, 4], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_1'], 'layer_2', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_2'], 'layer_3', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_3'], 'layer_4', 'TransposedConvolution', {'n_out_channels': 512, 'kernel_shape': [5, 5], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_4'], 'layer_5', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_5'], 'layer_6', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_6'], 'layer_7', 'TransposedConvolution', {'n_out_channels': 256, 'kernel_shape': [5, 5], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_7'], 'layer_8', 'Crop', {'crop_size_structure': [(2, 1), (2, 1)]}),
                (['layer_8'], 'layer_9', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_9'], 'layer_10', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_10'], 'layer_11', 'TransposedConvolution', {'n_out_channels': n_out_channels, 'kernel_shape': [5, 5], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_11'], 'layer_12', 'Crop', {'crop_size_structure': [(1, 2), (1, 2)]}),
                (['layer_12'], 'layer_13', 'ElementwiseApply', {'func': tf.nn.sigmoid}),
               ]
    if type == 'Critic' and image_size == [32, 32]:
        return [
                (['input_node'], 'layer_1', 'ConstantScaleShift', {'scale': 2., 'shift': -1.}),
                (['layer_1'], 'layer_2', 'Convolution', {'n_out_channels': 128, 'kernel_shape': [4, 4], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_2'], 'layer_3', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_3'], 'layer_4', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_4'], 'layer_5', 'Convolution', {'n_out_channels': 256, 'kernel_shape': [4, 4], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_5'], 'layer_6', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_6'], 'layer_7', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_7'], 'layer_8', 'Convolution', {'n_out_channels': 512, 'kernel_shape': [3, 3], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_8'], 'layer_9', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_9'], 'layer_10', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_10'], 'layer_11', 'Convolution', {'n_out_channels': 1024, 'kernel_shape': [3, 3], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_11'], 'layer_12', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_12'], 'layer_13', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_13'], 'layer_14', 'Convolution', {'n_out_channels': 1, 'kernel_shape': [2, 2], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
               ]
    if type == 'Generator' and image_size == [64, 64]:
        return [   
                (['input_node'], 'layer_1', 'TransposedConvolution', {'n_out_channels': 1024, 'kernel_shape': [4, 4], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_1'], 'layer_2', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_2'], 'layer_3', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_3'], 'layer_4', 'TransposedConvolution', {'n_out_channels': 512, 'kernel_shape': [5, 5], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_4'], 'layer_5', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_5'], 'layer_6', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_6'], 'layer_7', 'TransposedConvolution', {'n_out_channels': 256, 'kernel_shape': [5, 5], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_7'], 'layer_8', 'Crop', {'crop_size_structure': [(1, 2), (1, 2)]}),
                (['layer_8'], 'layer_9', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_9'], 'layer_10', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_10'], 'layer_11', 'TransposedConvolution', {'n_out_channels': 128, 'kernel_shape': [5, 5], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_11'], 'layer_12', 'Crop', {'crop_size_structure': [(2, 1), (2, 1)]}),
                (['layer_12'], 'layer_13', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_13'], 'layer_14', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_14'], 'layer_15', 'TransposedConvolution', {'n_out_channels': n_out_channels, 'kernel_shape': [5, 5], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_15'], 'layer_16', 'Crop', {'crop_size_structure': [(1, 2), (1, 2)]}),
                (['layer_16'], 'layer_17', 'ElementwiseApply', {'func': tf.nn.sigmoid})
               ]
    if type == 'Critic' and image_size == [64, 64]:
        return [
                (['input_node'], 'layer_1', 'ConstantScaleShift', {'scale': 2., 'shift': -1.}),
                (['layer_1'], 'layer_2', 'Convolution', {'n_out_channels': 128, 'kernel_shape': [5, 5], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_2'], 'layer_3', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_3'], 'layer_4', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_4'], 'layer_5', 'Convolution', {'n_out_channels': 256, 'kernel_shape': [5, 5], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_5'], 'layer_6', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_6'], 'layer_7', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_7'], 'layer_8', 'Convolution', {'n_out_channels': 512, 'kernel_shape': [5, 5], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_8'], 'layer_9', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_9'], 'layer_10', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_10'], 'layer_11', 'Convolution', {'n_out_channels': 1024, 'kernel_shape': [5, 5], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_11'], 'layer_12', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_12'], 'layer_13', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_13'], 'layer_14', 'Convolution', {'n_out_channels': 1, 'kernel_shape': [5, 5], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
               ]


#this removes batchnorm from critic as specified in wGAN-GP: https://arxiv.org/pdf/1704.00028.pdf
def get_dcgan_architectures_wgan(type='Generator', image_size=[28, 28], n_out_channels=1):
    if type == 'Generator' and image_size == [28, 28]:
        return [
                (['input_node'], 'layer_1', 'TransposedConvolution', {'n_out_channels': 1024, 'kernel_shape': [4, 4], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': False, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_1'], 'layer_2', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_2'], 'layer_3', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_3'], 'layer_4', 'TransposedConvolution', {'n_out_channels': 512, 'kernel_shape': [5, 5], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': False, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_4'], 'layer_5', 'Crop', {'crop_size_structure': [(0, 1), (0, 1)]}),
                (['layer_5'], 'layer_6', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_6'], 'layer_7', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_7'], 'layer_8', 'TransposedConvolution', {'n_out_channels': 256, 'kernel_shape': [5, 5], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': False, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_8'], 'layer_9', 'Crop', {'crop_size_structure': [(2, 1), (2, 1)]}),
                (['layer_9'], 'layer_10', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_10'], 'layer_11', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_11'], 'layer_12', 'TransposedConvolution', {'n_out_channels': n_out_channels, 'kernel_shape': [5, 5], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': False, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_12'], 'layer_13', 'Crop', {'crop_size_structure': [(1, 2), (1, 2)]}),
                (['layer_13'], 'layer_14', 'ElementwiseApply', {'func': tf.nn.sigmoid}),
               ]
    if type == 'Critic' and image_size == [28, 28]:
        return [
                (['input_node'], 'layer_1', 'ConstantScaleShift', {'scale': 2., 'shift': -1.}),
                (['layer_1'], 'layer_2', 'Convolution', {'n_out_channels': 128, 'kernel_shape': [3, 3], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_2'], 'layer_3', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_3'], 'layer_4', 'Convolution', {'n_out_channels': 256, 'kernel_shape': [3, 3], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_4'], 'layer_5', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_5'], 'layer_6', 'Convolution', {'n_out_channels': 512, 'kernel_shape': [3, 3], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_6'], 'layer_7', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_7'], 'layer_8', 'Convolution', {'n_out_channels': 1024, 'kernel_shape': [3, 3], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_8'], 'layer_9', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_9'], 'layer_10', 'Convolution', {'n_out_channels': 1, 'kernel_shape': [2, 2], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
               ]
    if type == 'Generator' and image_size == [32, 32]:
        return [   
                (['input_node'], 'layer_1', 'TransposedConvolution', {'n_out_channels': 1024, 'kernel_shape': [4, 4], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_1'], 'layer_2', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_2'], 'layer_3', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_3'], 'layer_4', 'TransposedConvolution', {'n_out_channels': 512, 'kernel_shape': [5, 5], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_4'], 'layer_5', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_5'], 'layer_6', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_6'], 'layer_7', 'TransposedConvolution', {'n_out_channels': 256, 'kernel_shape': [5, 5], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_7'], 'layer_8', 'Crop', {'crop_size_structure': [(2, 1), (2, 1)]}),
                (['layer_8'], 'layer_9', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_9'], 'layer_10', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_10'], 'layer_11', 'TransposedConvolution', {'n_out_channels': n_out_channels, 'kernel_shape': [5, 5], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_11'], 'layer_12', 'Crop', {'crop_size_structure': [(1, 2), (1, 2)]}),
                (['layer_12'], 'layer_13', 'ElementwiseApply', {'func': tf.nn.sigmoid}),
               ]
    if type == 'Critic' and image_size == [32, 32]:
        return [
                (['input_node'], 'layer_1', 'ConstantScaleShift', {'scale': 2., 'shift': -1.}),
                (['layer_1'], 'layer_2', 'Convolution', {'n_out_channels': 128, 'kernel_shape': [4, 4], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_2'], 'layer_3', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_3'], 'layer_4', 'Convolution', {'n_out_channels': 256, 'kernel_shape': [4, 4], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_4'], 'layer_5', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_5'], 'layer_6', 'Convolution', {'n_out_channels': 512, 'kernel_shape': [3, 3], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_6'], 'layer_7', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_7'], 'layer_8', 'Convolution', {'n_out_channels': 1024, 'kernel_shape': [3, 3], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_8'], 'layer_9', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_9'], 'layer_10', 'Convolution', {'n_out_channels': 1, 'kernel_shape': [2, 2], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
               ]
    if type == 'Generator' and image_size == [64, 64]:
        return [   
                (['input_node'], 'layer_1', 'TransposedConvolution', {'n_out_channels': 1024, 'kernel_shape': [4, 4], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_1'], 'layer_2', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_2'], 'layer_3', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_3'], 'layer_4', 'TransposedConvolution', {'n_out_channels': 512, 'kernel_shape': [5, 5], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_4'], 'layer_5', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_5'], 'layer_6', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_6'], 'layer_7', 'TransposedConvolution', {'n_out_channels': 256, 'kernel_shape': [5, 5], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_7'], 'layer_8', 'Crop', {'crop_size_structure': [(1, 2), (1, 2)]}),
                (['layer_8'], 'layer_9', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_9'], 'layer_10', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_10'], 'layer_11', 'TransposedConvolution', {'n_out_channels': 128, 'kernel_shape': [5, 5], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_11'], 'layer_12', 'Crop', {'crop_size_structure': [(2, 1), (2, 1)]}),
                (['layer_12'], 'layer_13', 'BatchNorm', {'mode': 'Regular'}),
                (['layer_13'], 'layer_14', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_14'], 'layer_15', 'TransposedConvolution', {'n_out_channels': n_out_channels, 'kernel_shape': [5, 5], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_15'], 'layer_16', 'Crop', {'crop_size_structure': [(1, 2), (1, 2)]}),
                (['layer_16'], 'layer_17', 'ElementwiseApply', {'func': tf.nn.sigmoid})
               ]
    if type == 'Critic' and image_size == [64, 64]:
        return [
                (['input_node'], 'layer_1', 'ConstantScaleShift', {'scale': 2., 'shift': -1.}),
                (['layer_1'], 'layer_2', 'Convolution', {'n_out_channels': 128, 'kernel_shape': [5, 5], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_2'], 'layer_3', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_3'], 'layer_4', 'Convolution', {'n_out_channels': 256, 'kernel_shape': [5, 5], 'strides': [2, 2], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_4'], 'layer_5', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_5'], 'layer_6', 'Convolution', {'n_out_channels': 512, 'kernel_shape': [5, 5], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_6'], 'layer_7', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_7'], 'layer_8', 'Convolution', {'n_out_channels': 1024, 'kernel_shape': [5, 5], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
                (['layer_8'], 'layer_9', 'ElementwiseApply', {'func': helper.LeakyReLU}),
                (['layer_9'], 'layer_10', 'Convolution', {'n_out_channels': 1, 'kernel_shape': [5, 5], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
               ]
