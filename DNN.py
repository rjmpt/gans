#this file contains custom low level layer definitons that I use in all my networks
#this file also contains functionality for building DNN objects from these custom layer definitions using proto architecture definitions as is done in architectures.py
#finally, there is extensive logging functionality that I have built in to this class in order to make creating new architectures fast and straightforward
#every layer has an __init__() method which is used to initially construct it, and then a forward() method in which an input tensor x is passed through the layer

from __future__ import print_function
from IPython.core.debugger import Pdb
pdb = Pdb()
trace = pdb.set_trace

import numpy as np
np.set_printoptions(suppress=True)
import tensorflow as tf
if tf.__version__ == '1.14.0':
    tf_compat_v1 = tf.compat.v1
    tf_compat_v1.enable_resource_variables()
else:
    tf_compat_v1 = tf

import helper

#superclass for all the various layers
class NNComponent():
    def __init__(self, stage='Training', name='NNComponent', tags=[]):
        assert (stage == 'Training' or stage == 'Test')
        
        self.tags = []
        for e in tags: 
            assert (type(e) is str)
            self.tags.append(e)

        self.stage = stage
        self.name = name

    def set_stage(self, stage):
        assert (stage == 'Training' or stage == 'Test')
        self.stage = stage
    
    def set_name(self, name):
        self.name = name

    def add_tag(self, tag):
        self.tags.append(tag)

    def remove_tag(self, tag):
        self.tags.remove(tag)

    def tag_signature(self):
        tag_signature_str = ''
        for e in self.tags:
            tag_signature_str += e + '--'
        return tag_signature_str[:-2]

    def n_params(self):
        return 0

    def generate_output_input_string(self, list_of_multiply_add_divide_lists):
        result_str = ''
        for i, multiply_add_divide_list in enumerate(list_of_multiply_add_divide_lists):
            multiply_number, add_number, divide_number = multiply_add_divide_list
            add_number_signed_str = str(add_number) if add_number<0 else '+'+str(add_number)
            result_str += 'out['+str(i)+'] = ['+'('+str(multiply_number)+'*in['+str(i)+']'+add_number_signed_str+')/'+str(divide_number)+']'+'\n'
        return result_str[:-1]

    def output_input_size_relation_list(self): 
        relation_list = [[1, 0, 1], [1, 0, 1]]
        return relation_list

    def output_input_size_relation(self, mode='string'):
        assert (mode == 'string' or mode == 'list')
        if mode == 'string': return self.generate_output_input_string(self.output_input_size_relation_list())
        else: return self.output_input_size_relation_list()

    def output_size_for_input_size(self, input_size):
        assert (input_size[0] > 0 and input_size[1] > 0)
        relation_list = self.output_input_size_relation('list')
        output_size = []
        for i in range(len(input_size)):
            try:
                multiply_number, add_number, divide_number = relation_list[i]
            except:
                trace()

            assert(divide_number > 0)
            output_size.append(int(np.floor(float(multiply_number*input_size[i]+add_number)/float(divide_number))))
        return output_size

#enables a single bias term for convolutional layers
class InputFreeBias(NNComponent):
    def __init__(self, variable_shape, stage='Training', initializer_mode='Zeros', name='input_free_bias', tags=[]):
        super().__init__(stage, name, tags)
        self.tags.append(self.__class__.__name__)

        self.variable_shape = variable_shape
        self.initializer_mode = initializer_mode
        self.weight_bias = None
        assert (len(self.variable_shape) == 4)
        assert (self.initializer_mode == 'Zeros' or self.initializer_mode == 'Ones')
    
    def initialize(self):
        if self.initializer_mode == 'Zeros':
            initial_weight_bias = np.zeros(self.variable_shape, np.float32)
        elif self.initializer_mode == 'Ones':
            initial_weight_bias = np.ones(self.variable_shape, np.float32)
        self.weight_bias = tf_compat_v1.Variable(initial_weight_bias, name=self.tag_signature()+'/'+self.name+'_weight_bias', trainable=True)

    def forward(self, x):
        if self.bias is None: self.initialize()
        return self.weight_bias

#handles mean reduce, sum reduce, product reduce, and concatenation
class Reduce(NNComponent):
    def __init__(self, stage='Training',  mode='Sum', name='reduce', tags=[]):
        super().__init__(stage, name, tags)
        self.tags.append(self.__class__.__name__)

        self.mode = mode
        assert(mode == 'Mean' or mode == 'Sum' or mode == 'Product' or mode == 'Concat')

    def forward(self, x):
        if self.mode == 'Mean':
            return tf.add_n(x)/len(n)
        if self.mode == 'Sum':
            return tf.add_n(x)
        if self.mode == 'Product':
            out = 1
            for e in x: out *= e
            return out
        if self.mode == 'Concat':
            return tf.concat(x, axis=-1)

#element wise operations
class ElementwiseApply(NNComponent):
    def __init__(self, stage='Training', func=None, name='elementwise_apply', tags=[]):
        super().__init__(stage, name, tags)
        self.tags.append(self.__class__.__name__)

        self.func = func
        
    def forward(self, x):
        if self.func:
            out = self.func(x)
        else:
            out = x

        return out

#perform a*x + b operation to all outputs of another layer
class ConstantScaleShift(NNComponent):
    def __init__(self, stage='Training', scale=1., shift=0., name='constant_scale_shift', tags=[]):
        super().__init__(stage, name, tags)
        self.tags.append(self.__class__.__name__)
        self.scale = scale
        self.shift = shift
    
    def forward(self, x):
        out = x
        if self.scale != 1.:
            out = self.scale*out
        if self.shift != 0.:
            out = out+self.shift
        return out

#spatial dropout, ensures that the correct operations happen during train/test
class SpatialDropout(NNComponent):
    def __init__(self, stage='Training', dropout_rate=0.5, name='spatial_dropout', tags=[]):
        super().__init__(stage, name, tags)
        self.tags.append(self.__class__.__name__)

        self.dropout_rate = dropout_rate
        assert (self.dropout_rate >= 0 and self.dropout_rate <= 1)

    def forward(self, x):
        if self.stage == 'Training':
            dropout_mask = tf.stop_gradient(tf.cast(tf_compat_v1.random.uniform(shape=(tf.shape(x)[0], 1, 1, tf.shape(x)[-1])) < (1-self.dropout_rate), tf.float32))
            out = x * dropout_mask/(1-self.dropout_rate)
        elif self.stage == 'Test':
            out = x
        return out

#tensorflow equivalent of: https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html, useful when upsampling
class PixelShuffle(NNComponent):
    def __init__(self, upsample_rate=[1,1], stage='Training', name='pixel_shuffle', tags=[]):
        super().__init__(stage, name, tags)
        self.tags.append(self.__class__.__name__)

        self.upsample_rate = upsample_rate

    def output_input_size_relation_list(self):
        relation_list = [[1, self.upsample_rate[0], 1/(self.upsample_rate[0]*self.upsample_rate[1])], [1, self.upsample_rate[1],  1/(self.upsample_rate[0]*self.upsample_rate[1])]]
        return relation_list

    def get_np_grid(self, shape, i, f1, f2):
        i = i % (f1*f2)
        grid = np.zeros(shape,dtype=np.float32)
        r_start = i//f2
        c_start = i % f2

        x_list = list(range(c_start, grid.shape[2], f2))
        y_list = list(range(r_start, grid.shape[1], f1))

        x_list_new = x_list*len(y_list)
        y_list_new = []
        for y in y_list:
            y_list_new += [y]*len(x_list)

        grid[:, y_list_new, x_list_new] = 1
        return grid

    def forward(self, x):
        [f1, f2] = self.upsample_rate
        [b, h, w, c] = x.shape

        assert (c % (f1*f2) == 0)

        x_resized = tf_compat_v1.keras.backend.repeat_elements(x[:,:,:,0], f1, axis=1)
        x_resized = tf_compat_v1.keras.backend.repeat_elements(x_resized, f2, axis=2)
        current_slice = tf.zeros_like(x_resized)

        np_grid_shape = [1,h*f1,w*f2]

        slice_list = []

        for i in range(c):
           tiled_slice = tf_compat_v1.keras.backend.repeat_elements(x[:,:,:,i], f1, axis=1)
           tiled_slice = tf_compat_v1.keras.backend.repeat_elements(tiled_slice, f2, axis=2)
           tiled_slice = tiled_slice * tf.constant(self.get_np_grid(np_grid_shape, i, f1, f2))

           if i > 0 and i%(f1*f2) == 0:
               slice_list += [current_slice]
               current_slice = tiled_slice
           else:
               current_slice += tiled_slice


        slice_list += [current_slice]
        return tf.stack(slice_list, axis=3)

#standard upsampling (nearest neighbor)
class Upsample(NNComponent):
    def __init__(self, stage='Training', upsample_rate=[2, 2], mode='Nearest Neighbor', name='upsample', tags=[]):
        super().__init__(stage, name, tags)
        self.tags.append(self.__class__.__name__)

        self.upsample_rate = upsample_rate
        self.mode = mode
        assert (self.mode == 'Nearest Neighbor')
        assert (self.upsample_rate[0] > 0 and self.upsample_rate[1] > 0)

    def output_input_size_relation_list(self):
        relation_list = [[1, self.upsample_rate[0], 1], [1, self.upsample_rate[1], 1]]
        return relation_list

    def forward(self, x):
        if self.mode == 'Nearest Neighbor':
            return helper.tf_nn_upsample_tensor(x, upsample_rate=self.upsample_rate)

#subsample activations based on the specified stride structure
class Subsample(NNComponent):
    def __init__(self, subsample_structure=[(1, 1), (0, 0)], stage='Training', name='subsample', tags=[]):
        super().__init__(stage, name, tags)
        self.tags.append(self.__class__.__name__)

        self.stride = subsample_structure[0]
        self.stride_ind = subsample_structure[1]
        assert (self.stride_ind[0] < self.stride[0] and self.stride_ind[1] < self.stride[1])

    def output_input_size_relation_list(self):
        relation_list = [[1, 0, self.stride[0]], [1, 0, self.stride[1]]]
        return relation_list

    def forward(self, x):
        x_subsampled = x
        if self.stride[0] != 1 or self.stride[1] != 1:
            x_cropped = x[:, :self.stride[0]*(tf.shape(x)[1]//self.stride[0]), 
                                     :self.stride[1]*(tf.shape(x)[2]//self.stride[1]), :]
            x_subsampled = x_cropped[:, self.stride_ind[0]::self.stride[0], self.stride_ind[1]::self.stride[1], :] 
        
        return x_subsampled

#crop using the specified crop structure [(top, bottom), (left, right)]
class Crop(NNComponent):
    def __init__(self, crop_size_structure=[(1, 1), (1, 1)], stage='Training', name='crop', tags=[]):
        super().__init__(stage, name, tags)
        self.tags.append(self.__class__.__name__)

        self.crop_size_structure = crop_size_structure

    def output_input_size_relation_list(self):
        relation_list = [[1, -self.crop_size_structure[0][0]-self.crop_size_structure[0][1], 1], 
                         [1, -self.crop_size_structure[1][0]-self.crop_size_structure[1][1], 1]]
        return relation_list

    def get_hw_indices(self, x):
        if x.get_shape()[1].value is None: h_size = tf.shape(x)[1]
        else: h_size = x.get_shape()[1].value
        if x.get_shape()[2].value is None: w_size = tf.shape(x)[2]
        else: w_size = x.get_shape()[2].value

        h_ind_start, h_ind_end = self.crop_size_structure[0][0], h_size-self.crop_size_structure[0][1]
        w_ind_start, w_ind_end = self.crop_size_structure[1][0], w_size-self.crop_size_structure[1][1]
        return h_ind_start, h_ind_end, w_ind_start, w_ind_end

    def forward(self, x):
        x_cropped = x
        if self.crop_size_structure[0][0] != 0 or self.crop_size_structure[0][1] or \
           self.crop_size_structure[1][0] != 0 or self.crop_size_structure[1][1]:
            h_ind_start, h_ind_end, w_ind_start, w_ind_end = self.get_hw_indices(x)
            x_cropped = x[:, h_ind_start:h_ind_end, w_ind_start:w_ind_end, :]
        
        return x_cropped

#standard pooling layer, superclass for Pooling layer which implements avg and max pooling
class GeneralPooling(NNComponent):
    def __init__(self, stage='Training', kernel_shape=[2, 2], strides=[2, 2], dilations=[1, 1], force_explicit_crop=False, crop_mode='center', name='GeneralPooling', tags=[]):
        super().__init__(stage, name, tags)
        self.kernel_shape = kernel_shape
        self.strides = strides
        self.dilations = dilations
        self.dilated_kernel_shape = [self.kernel_shape[0]+(self.dilations[0]-1)*(self.kernel_shape[0]-1),
                                     self.kernel_shape[1]+(self.dilations[1]-1)*(self.kernel_shape[1]-1)]
        self.padding = "VALID"
        self.force_explicit_crop = force_explicit_crop
        self.crop_mode = crop_mode
        self.crop_size_structure = None

        assert (self.kernel_shape[0] > 0 and self.kernel_shape[1] > 0)
        assert (self.strides[0] > 0 and self.strides[1] > 0)
        assert (self.dilations[0] > 0 and self.dilations[1] > 0)
        assert ((self.dilations[0] == 1 and self.dilations[1] == 1) or (self.strides[1] == 1 and self.strides[1] == 1))

    def output_input_size_relation_list(self):
        relation_list = [[1, -(self.dilated_kernel_shape[0]-self.strides[0]), self.strides[0]], 
                         [1, -(self.dilated_kernel_shape[1]-self.strides[1]), self.strides[1]]]
        return relation_list

    def get_crop_shape(self, input_shape):
        if self.force_explicit_crop:
            x_crop = (input_shape[0] - self.dilated_kernel_shape[0]) % self.strides[0]
            y_crop = (input_shape[1] - self.dilated_kernel_shape[1]) % self.strides[1]
        
            if self.crop_mode == 'center':
                #if the crop is odd, crop more from the top left
                x_crop_right = x_crop // 2
                y_crop_bottom = y_crop // 2

                return [(x_crop - x_crop_right, x_crop_right), (y_crop - y_crop_bottom, y_crop_bottom)]
            
            else:
                x_crop_tuple = None
                y_crop_tuple = None
                
                if 'left' in self.crop_mode:
                    x_crop_tuple = (x_crop, 0)
                elif 'right' in self.crop_mode:
                    x_crop_tuple = (0, x_crop)
                if 'top' in self.crop_mode:
                    y_crop_tuple = (y_crop, 0)
                elif 'bottom' in self.crop_mode:
                    y_crop_tuple = (0, y_crop)

                assert (x_crop_tuple and y_crop_tuple)
                return [x_crop_tuple, y_crop_tuple]

        else:
            return [(0, (input_shape[0] - self.dilated_kernel_shape[0]) % self.strides[0]), 
                                            (0, (input_shape[1] - self.dilated_kernel_shape[1]) % self.strides[1])]

    def explicit_crop(self, x):
        return Crop(self.crop_size_structure).forward(x)

#avg and max pooling
class Pooling(GeneralPooling):
    def __init__(self, stage='Training', kernel_shape=[2, 2], strides=[2, 2], dilations=[1, 1], pooling_type='MAX', force_explicit_crop=False, crop_mode='center', name='pooling', tags=[]):
        super().__init__(stage=stage, kernel_shape=kernel_shape, strides=strides, dilations=dilations, force_explicit_crop=force_explicit_crop, crop_mode=crop_mode, name=name, tags=tags)
        self.tags.append(self.__class__.__name__)
        self.pooling_type = pooling_type
        assert (self.pooling_type == 'AVG' or self.pooling_type == 'MAX')

    def forward(self, x):
        self.crop_size_structure = self.get_crop_shape(x.get_shape().as_list()[1:3])
        if self.force_explicit_crop: 
            x = self.explicit_crop(x)
            trace()

        out = tf.nn.pool(input=x, window_shape=self.kernel_shape, pooling_type=self.pooling_type, strides=self.strides, padding='VALID', data_format='NHWC', dilation_rate=self.dilations)
        return out

#superclass for all convolution types
class GeneralConvolution(GeneralPooling):
    def __init__(self, stage, initializer_mode, n_out_channels, kernel_shape, strides, dilations, force_explicit_crop, crop_mode, use_bias, force_no_matmul, name, tags):
        super().__init__(stage=stage, kernel_shape=kernel_shape, strides=strides, dilations=dilations, force_explicit_crop=force_explicit_crop, crop_mode=crop_mode, name=name, tags=tags)        
        self.initializer_mode = initializer_mode
        self.n_out_channels = n_out_channels
        self.use_bias = use_bias
        self.weight_bias = None
        self.weight_kernel = None
        self.force_no_matmul = force_no_matmul
    
        assert (self.strides[0] <= self.dilated_kernel_shape[0] and self.strides[1] <= self.dilated_kernel_shape[1])

    def add_tag(self, tag):
        try: assert (self.weight_bias == None and self.weight_kernel == None)
        except: print('Tagging failed. The parameters of the component are already initialized.'); quit()
        self.tags.append(tag)

    def remove_tag(self, tag):
        try: assert (self.weight_bias == None and self.weight_kernel == None)
        except: print('Tagging failed. The parameters of the component are already initialized.'); quit()
        self.tags.remove(tag)

    def get_initial_weight_bias_np(self, n_out_channels):
        initial_weight_bias = np.zeros([1, 1, 1, n_out_channels], np.float32)
        return initial_weight_bias

    def get_initial_weight_kernel_np(self, n_in_channels, n_out_channels):
        # Written with convolution in mind, using terminology related to convolutions.
        fan_in, fan_out = n_in_channels*np.prod(self.kernel_shape), n_out_channels*np.prod(self.kernel_shape)
        if 'uniform' in self.initializer_mode:
            if self.initializer_mode == 'uniform':
                uniform_scale = 0.05
            elif self.initializer_mode == 'lecun_uniform': # http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
                uniform_scale = np.sqrt(3./fan_in)
            elif self.initializer_mode == 'xavier_uniform': # http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
                uniform_scale = np.sqrt(6./(fan_in + fan_out))
            elif self.initializer_mode == 'he_uniform': # http://arxiv.org/abs/1502.01852
                uniform_scale = np.sqrt(6./fan_in)
            initial_weight_kernel = np.random.uniform(low=-uniform_scale, high=uniform_scale, size=self.kernel_shape+[n_in_channels, n_out_channels]).astype(np.float32)

        elif 'gaussian' in self.initializer_mode:
            if self.initializer_mode == 'gaussian' or self.initializer_mode == 'truncated_gaussian':
                gaussian_std = 0.02
            elif self.initializer_mode == 'xavier_gaussian': # http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
                gaussian_std = np.sqrt(2./(fan_in+fan_out))
            elif self.initializer_mode == 'he_gaussian': # http://arxiv.org/abs/1502.01852
                gaussian_std = np.sqrt(2./fan_in)

            if 'truncated' in self.initializer_mode:
                initial_weight_kernel = helper.truncated_gaussian_sampler_np(self.kernel_shape+[n_in_channels, n_out_channels], mu=0, std=gaussian_std, lower=None, upper=None).astype(np.float32)
            else: 
                initial_weight_kernel = np.random.randn(*(self.kernel_shape+[n_in_channels, n_out_channels])).astype(np.float32)*gaussian_std

        return initial_weight_kernel

    def n_params(self):
        if self.weight_kernel is None:
            print('Parameter sizes are not set, as the number of input channels is unknown. Requires forwarding an input tensor.')
            quit()
        else:
            n_params = np.prod(self.weight_kernel.get_shape().as_list())
            if self.use_bias: n_params += np.prod(self.weight_bias.get_shape().as_list())
            return n_params

#typical convolutional layer, can handle any initialization modes, kernel shapes, strides, dilations
class Convolution(GeneralConvolution):
    def __init__(self, stage='Training', initializer_mode='xavier_uniform', n_out_channels=32, kernel_shape=[3, 3], strides=[1, 1], dilations=[1, 1], use_bias=True, force_no_matmul=False, force_explicit_crop=False, crop_mode='center', padding_mode='VALID', name='convolution', tags=[]):
        super().__init__(stage, initializer_mode, n_out_channels, kernel_shape, strides, dilations, force_explicit_crop, crop_mode, use_bias, force_no_matmul, name, tags)
        self.tags.append(self.__class__.__name__)
        self.padding_mode = padding_mode
        self.pad_size_structure = self.get_pad_shape()

        assert (self.padding_mode == 'VALID' or (self.padding_mode == 'SAME' and self.strides == [1,1])) #SAME padding makes no sense is conv stride > 1

    def initialize(self, n_in_channels):
        init_weight_kernel_np = self.get_initial_weight_kernel_np(n_in_channels=n_in_channels, n_out_channels=self.n_out_channels)
        self.weight_kernel = tf_compat_v1.Variable(init_weight_kernel_np, name=self.tag_signature()+'/'+self.name+'_weight_kernel', trainable=True)

        init_weight_bias_np = self.get_initial_weight_bias_np(n_out_channels=self.n_out_channels)
        self.weight_bias = tf_compat_v1.Variable(init_weight_bias_np, name=self.tag_signature()+'/'+self.name+'_weight_bias', trainable=True)
    
    def output_input_size_relation_list(self):
        if self.padding_mode == 'VALID':
            relation_list = [[1, -(self.dilated_kernel_shape[0]-self.strides[0]), self.strides[0]], 
                            [1, -(self.dilated_kernel_shape[1]-self.strides[1]), self.strides[1]]]
        else:
            relation_list = [[1, 0, self.strides[0]], [1, 0, self.strides[1]]]

        return relation_list

    def forward(self, x):
        if self.weight_kernel is None: self.initialize(n_in_channels=x.get_shape()[3].value)        
        assert (x.get_shape()[3].value == self.weight_kernel.get_shape()[-2].value)
        
        if (x.get_shape().as_list()[1:3] == self.kernel_shape) and (self.dilations == [1, 1]) and (not self.force_no_matmul):
            self.crop_size_structure = [(0, 0), (0, 0)]
            x_flattened = tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:4])])

            weight_kernel_flattened = tf.reshape(self.weight_kernel, [np.prod(self.weight_kernel.get_shape().as_list()[:3]), -1])
            out = tf.matmul(x_flattened, weight_kernel_flattened)
            if self.use_bias: out = tf.nn.bias_add(out, self.weight_bias[0, 0, 0, :])
            out = out[:, np.newaxis, np.newaxis, :]
        else:
            self.crop_size_structure = self.get_crop_shape(x.get_shape().as_list()[1:3])
            if self.force_explicit_crop: 
                x = self.explicit_crop(x)
                trace()
            out = tf.nn.conv2d(input=x, filter=self.weight_kernel, strides=[1, self.strides[0], self.strides[1], 1], dilations=[1, self.dilations[0], self.dilations[1], 1], padding=self.padding_mode, data_format='NHWC')
            if self.use_bias: out = tf.nn.bias_add(out, self.weight_bias[0, 0, 0,:], data_format='NHWC')

        return out
    
    def get_pad_shape(self):
        if self.padding_mode == 'VALID':
            return [(0,0), (0,0)]

        x_pad = self.dilated_kernel_shape[0] - 1
        y_pad = self.dilated_kernel_shape[1] - 1
     
        #if the padding is odd, pad more to the bottom right
        x_pad_left = x_pad // 2
        y_pad_top = y_pad // 2
    
        return [(x_pad_left, x_pad-x_pad_left), (y_pad_top, y_pad-y_pad_top)]
    
#transposed convolutional layer for learned upsampling    
class TransposedConvolution(GeneralConvolution):
    def __init__(self, stage='Training', initializer_mode='xavier_uniform', n_out_channels=32, kernel_shape=[3, 3], strides=[1, 1], dilations=[1, 1], use_bias=True, force_no_matmul=False, name='transposed_convolution', tags=[]):
        super().__init__(stage, initializer_mode, n_out_channels, kernel_shape, strides, dilations, 'False', 'center', use_bias, force_no_matmul, name, tags)
        self.tags.append(self.__class__.__name__)
        # Describes the relation between tf.nn.conv2d and tf.nn.conv2d_transpose: https://github.com/tensorflow/tensorflow/issues/2118

    def initialize(self, n_in_channels, mirror_initialization=True):
        # weight_kernel shape structure: [height x width x self.n_out_channels x self.n_in_channels], last two dims. swapped compared to the convolution kernel.
        if mirror_initialization: init_weight_kernel_np = self.get_initial_weight_kernel_np(n_in_channels=self.n_out_channels, n_out_channels=n_in_channels)
        else: init_weight_kernel_np = np.transpose(self.get_initial_weight_kernel_np(n_in_channels=n_in_channels, n_out_channels=self.n_out_channels), [0, 1, 3, 2])
        self.weight_kernel = tf_compat_v1.Variable(init_weight_kernel_np, name=self.tag_signature()+'/'+self.name+'_weight_kernel', trainable=True)
        
        init_weight_bias_np = self.get_initial_weight_bias_np(n_out_channels=self.n_out_channels)
        self.weight_bias = tf_compat_v1.Variable(init_weight_bias_np, name=self.tag_signature()+'/'+self.name+'_weight_bias', trainable=True)

    def output_input_size_relation_list(self):
        relation_list = [[self.strides[0], (self.dilated_kernel_shape[0]-self.strides[0]), 1], 
                         [self.strides[1], (self.dilated_kernel_shape[1]-self.strides[1]), 1]]
        return relation_list

    def forward(self, x):
        if self.weight_kernel is None: self.initialize(n_in_channels=x.get_shape()[3].value)
        assert (x.get_shape()[3].value == self.weight_kernel.get_shape()[-1].value)
        
        if (x.get_shape().as_list()[1:3] == [1, 1]) and (self.dilations == [1, 1]) and (not self.force_no_matmul):
            output_shape = [tf.shape(x)[0], self.kernel_shape[0], self.kernel_shape[1], self.n_out_channels]
            x_flattened = tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:4])])

            weight_kernel_flattened = tf.transpose(tf.reshape(self.weight_kernel, [np.prod(self.weight_kernel.get_shape().as_list()[:3]), -1]), [1, 0])
            out = tf.matmul(x_flattened, weight_kernel_flattened)
            out = tf.reshape(out, output_shape)                
        else:
            output_shape = [tf.shape(x)[0], x.get_shape()[1].value*self.strides[0]+self.dilated_kernel_shape[0]-self.strides[0], x.get_shape()[2].value*self.strides[1] + self.dilated_kernel_shape[1]-self.strides[1] , self.n_out_channels]
            out = tf.nn.conv2d_transpose(input=x, filter=self.weight_kernel, output_shape=output_shape, strides=[1, self.strides[0], self.strides[1], 1], dilations=[1, self.dilations[0], self.dilations[1], 1], padding="VALID", data_format='NHWC')
        
        if self.use_bias: out = tf.nn.bias_add(out, self.weight_bias[0, 0, 0, :], data_format='NHWC')

        return out

#batchnorm layer
class BatchNorm(NNComponent):
    def __init__(self, stage='Training', mode='Regular', exponential_decay=0.01, epsilon=0.001, d_max=5, r_max=3, name='batch_norm', tags=[]):
        super().__init__(stage, name, tags)
        self.tags.append(self.__class__.__name__)

        self.mode = mode
        self.exponential_decay = exponential_decay
        self.epsilon = epsilon
        self.d_max = d_max
        self.r_max = r_max

        assert(mode == 'Regular' or mode == 'Renormalization')
        # see below for the Renormalization option
        # https://papers.nips.cc/paper/6790-batch-renormalization-towards-reducing-minibatch-dependence-in-batch-normalized-models.pdf
        self.batch_acc_mu = None
        self.batch_acc_var = None
        self.global_mu = None
        self.global_std = None

    def n_params(self):
        if self.batch_acc_mu is None:
            print('Parameter sizes are not set. Requires forwarding an input tensor.')
            quit()
        else:
            return self.batch_acc_mu.shape()[-1].value+self.batch_acc_var.shape()[-1].value+self.global_mu.shape()[-1].value+self.global_std.shape()[-1].value

    def add_tag(self, tag):
        assert (self.batch_acc_mu == None and self.batch_acc_var == None and self.global_mu == None and self.global_std == None)
        self.tags.append(tag)

    def remove_tag(self, tag):
        assert (self.batch_acc_mu == None and self.batch_acc_var == None and self.global_mu == None and self.global_std == None)
        self.tags.remove(tag)

    def initialize(self, n_in_channels): 
        self.batch_acc_mu = tf_compat_v1.Variable(tf.zeros((1, 1, 1, n_in_channels), tf.float32), name=self.tag_signature()+'/'+self.name+'_bn_acc_mu_0', trainable=True)
        self.batch_acc_var = tf_compat_v1.Variable(tf.ones((1, 1, 1, n_in_channels), tf.float32), name=self.tag_signature()+'/'+self.name+'_bn_acc_var_0', trainable=True)
        self.global_mu = tf_compat_v1.Variable(tf.zeros((1, 1, 1, n_in_channels), tf.float32), name=self.tag_signature()+'/'+self.name+'_bn_global_mu_0', trainable=True)
        self.global_std = tf_compat_v1.Variable(tf.ones((1, 1, 1, n_in_channels), tf.float32), name=self.tag_signature()+'/'+self.name+'_bn_global_std_0', trainable=True)
        
    def forward(self, x):
        if self.batch_acc_mu is None:
            self.initialize(n_in_channels=x.shape[-1].value)

        if self.stage == 'Training':
            batch_mu = tf.reduce_mean(x, axis=[0, 1, 2], keepdims=True) # [1 x 1 x 1 x channel]
            batch_var = tf.reduce_mean((x-batch_mu)**2, axis=[0, 1, 2], keepdims=True) # [1 x 1 x 1 x channel]
            batch_acc_mu_update_op = tf_compat_v1.assign(self.batch_acc_mu, (1-self.exponential_decay)*self.batch_acc_mu+(self.exponential_decay)*batch_mu)
            batch_acc_var_update_op = tf_compat_v1.assign(self.batch_acc_var, (1-self.exponential_decay)*self.batch_acc_var+(self.exponential_decay)*batch_var)

            with tf.control_dependencies([batch_acc_mu_update_op, batch_acc_var_update_op]):
                if self.mode == 'Regular':
                    out = ((x - batch_mu)/helper.safe_tf_sqrt(self.epsilon+batch_var))*self.global_std+self.global_mu
                elif self.mode == 'Renormalization':
                    r = tf.stop_gradient(tf.clip_by_value(helper.safe_tf_sqrt(self.epsilon+batch_var)/helper.safe_tf_sqrt(self.epsilon+batch_acc_var), 1/self.r_max, self.r_max))
                    d = tf.stop_gradient(tf.clip_by_value((batch_mu-self.batch_acc_mu)/helper.safe_tf_sqrt(self.epsilon+batch_acc_var), -self.d_max, self.d_max))
                    out = (((x - batch_mu)/helper.safe_tf_sqrt(self.epsilon+batch_var))*r+d)*self.global_std+self.global_mu
        elif self.stage == 'Test':
            out = ((x - self.batch_acc_mu)/helper.safe_tf_sqrt(self.epsilon+self.batch_acc_var))*self.global_std+self.global_mu
        return out

#used to construct DNN objects which are Directed Acyclic Graphs (DAGs) of the various NN Components
class DNN():
    def __init__(self, structure_list=None, proto_structure_list=None, name='DNN'):
        self.name = name
        assert ((structure_list is None and proto_structure_list is not None) or (structure_list is not None and proto_structure_list is None))

        if structure_list is None:
            self.structure_list = []
            for e in proto_structure_list:
                assert (type(e[0]) is list and type(e[1]) is str)
                assert (type(e[2]) is str and type(e[3]) is dict)
                self.structure_list.append((e[0], e[1], globals()[e[2]](**e[3])))

        else:
            self.structure_list = []
            for e in structure_list:
                if hasattr(e[2], 'get_structure_list'):
                    assert (len(e[0]) == 1) # has one input
                    e[2].set_structure_list(e[0][0], e[1])
                    self.structure_list = self.structure_list+(e[2].get_structure_list())
                else:
                    self.structure_list.append(e)

        # two assumptions on the self.structure_list, 1. incoming node is ['input_node'] and 2. the order in the
        # self.structure_list follows the natural order of computation, i.e. all incoming nodes at a particular point
        # in the list has already been computed in an earlier element as an output node. As a consequence, the last
        # element computes the final output. Layer names must be unique etc. common sense stuff.
        # e.g. element of this list [(['input_node'], 'layer_1', ValidConvolution object), (['layer_1'], 'layer_2', ValidConvolution object)]
        self.incoming_nodes_list = [e[0] for e in self.structure_list]
        self.output_nodes_list = [e[1] for e in self.structure_list]
        self.components_list = [e[2] for e in self.structure_list]

        self.set_names_and_tags()

        assert(len(self.structure_list) > 0)
        assert(all([input_list == ['input_node'] if isinstance(layer, InputFreeBias) else True for (input_list, output_node, layer) in self.structure_list]))

    def set_names_and_tags(self):
        for component, output_node in zip(self.components_list, self.output_nodes_list):
            component.set_name(output_node)
            if len(component.tags) > 0: component.add_tag(self.name)

    def set_stage(self, stage):
        for e in self.components_list:
            e.set_stage(stage)
    
    def n_params(self):
        n_params = 0
        for i, layer in enumerate(self.components_list):
            n_params += layer.n_params()
        return n_params

    def get_network_activation_sizes(self, input_size, verbose=3):
        assert (0 <= verbose <= 3)
        all_sizes = {'input_node' : input_size+[None]}
        
        for i, layer in enumerate(self.components_list):
            layer_verbose_info = ''

            if isinstance(layer, Convolution) or isinstance(layer, TransposedConvolution):                
                all_sizes[self.output_nodes_list[i]] = layer.output_size_for_input_size(all_sizes[self.incoming_nodes_list[i][0]][:2])+[layer.n_out_channels]
            elif isinstance(layer, Pooling) or isinstance(layer, Upsample) or isinstance(layer, Subsample) or isinstance(layer, Crop):
                all_sizes[self.output_nodes_list[i]] = layer.output_size_for_input_size(all_sizes[self.incoming_nodes_list[i][0]][:2])+[all_sizes[self.incoming_nodes_list[i][0]][2]]
            else:
                all_sizes[self.output_nodes_list[i]] = all_sizes[self.incoming_nodes_list[i][0]]

            if verbose >= 2:
                layer_verbose_info = 'Component Class: ' + layer.__class__.__name__
                if (isinstance(layer, Convolution) or isinstance(layer, TransposedConvolution) or isinstance(layer, Pooling)):
                    layer_verbose_info += ', Kernel Shape: ' + str(layer.kernel_shape) + ', Dilated Kernel Shape: ' + str(layer.dilated_kernel_shape) + ', Strides: ' + str(layer.strides)
                    if isinstance(layer, Convolution) or isinstance(layer, TransposedConvolution):
                        layer_verbose_info += ', Bias: ' + str(layer.use_bias)
                elif isinstance(layer, BatchNorm):
                    layer_verbose_info += ', Mode: ' + str(layer.mode)
                elif isinstance(layer, ElementwiseApply):
                    layer_verbose_info += ', Function Name: ' + str(layer.func.__name__)
                elif isinstance(layer, SpatialDropout):
                    layer_verbose_info += ', Dropout Rate: ' + str(layer.dropout_rate)
                elif isinstance(layer, Upsample):
                    layer_verbose_info += ', Upsample Rate: ' + str(upsample_rate)
                elif isinstance(layer, Subsample):
                    layer_verbose_info += ', Subsample Stucture: PERHAPS SOMETHING BETTER? ' + str(layer.subsample_structure)
                elif isinstance(layer, Crop):
                    [(crop_left, crop_right), (crop_top, crop_bottom)] = layer.crop_size_structure
                    layer_verbose_info += ', Crop Stucture: [Left: ' + str(crop_left) + ', Right: ' + str(crop_right) + ', Top: ' + str(crop_top) + ', Bottom: ' + str(crop_bottom) + ']'

                if verbose == 3:
                    if (isinstance(layer, Convolution) or isinstance(layer, Pooling)):
                        [(crop_left, crop_right), (crop_top, crop_bottom)] = layer.get_crop_shape(input_size)

                        if isinstance(layer, Convolution):
                            [(pad_left, pad_right), (pad_top, pad_bottom)] = layer.get_pad_shape()
                            op = '\nConv Op, '
                            if not layer.force_no_matmul and layer.dilations == [1,1] and all_sizes[self.incoming_nodes_list[i][0]][:2] == layer.kernel_shape:
                                op = '\nMatmul Op, '
                            layer_verbose_info += op 
                        
                            if layer.padding_mode == 'VALID':
                                layer_verbose_info += 'Crop Stucture: [Left: ' + str(crop_left) + ', Right: ' + str(crop_right) + ', Top: ' + str(crop_top) + ', Bottom: ' + str(crop_bottom) + ']'
                            else:
                                layer_verbose_info += 'Pad Stucture: [Left: ' + str(pad_left) + ', Right: ' + str(pad_right) + ', Top: ' + str(pad_top) + ', Bottom: ' + str(pad_bottom) + ']'

                        else:
                            layer_verbose_info += '\nCrop Stucture: [Left: ' + str(crop_left) + ', Right: ' + str(crop_right) + ', Top: ' + str(crop_top) + ', Bottom: ' + str(crop_bottom) + ']'
                            layer_verbose_info += '\n'


            current_output_name = self.output_nodes_list[i]
            if i == 0:
                layer_str = '\n\n\n**************************************************************************************************************************\n\t\t\t\t\t\t\t' 
                layer_str += 'DNN: ' + self.name + '\n**************************************************************************************************************************\n\n' + 'Incoming nodes: '
            else:
                layer_str = 'Incoming nodes: '
            for e in self.incoming_nodes_list[i]: layer_str = layer_str +' [' + e + ': '+ str(all_sizes[e]) + ']'
            layer_str = layer_str + ', Output('+current_output_name+'): ' + str(all_sizes[current_output_name])
            layer_str = layer_str + '\n' + layer_verbose_info
            if verbose > 0: print(layer_str+'\n\n'+'**************************************************************************************************************************'+'\n')
        if verbose > 0: print('\n\n\n')

        return all_sizes
        
    def output_input_size_relation_string(self):
        list_of_relation_lists = None
        for i, layer in enumerate(self.components_list):
            relation_str = layer.output_input_size_relation('string')
            relation_str_list = relation_str.split('\n')
            if list_of_relation_lists is None: list_of_relation_lists = [[] for e in relation_str_list]
            for i, e in enumerate(relation_str_list): list_of_relation_lists[i].append(e)

        full_relation = ''
        for dim, relation_list_curr_dim in enumerate(list_of_relation_lists):
            canonical_out_str = None
            overall_relation_str = ''
            for i, relation_str in enumerate(relation_list_curr_dim):
                out_str, input_str = relation_str.split(' = ')
                if i == 0:
                    canonical_out_str = out_str
                    overall_relation_str = input_str
                    if input_str == '[(1*'+'in['+str(dim)+']'+'+0)/1]': 
                        overall_relation_str = 'in['+str(dim)+']'
                    else: 
                        overall_relation_str = input_str
                else: 
                    assert (canonical_out_str == out_str)
                    input_str_left, input_str_right = input_str.split('in['+str(dim)+']')
                    if input_str_left != '[(1*' or input_str_right != '+0)/1]':
                        overall_relation_str = input_str_left+'('+overall_relation_str+')'+input_str_right

            overall_relation_str = canonical_out_str + ' = ' + overall_relation_str
            full_relation = full_relation + overall_relation_str + '\n'
        full_relation = full_relation[:-1]
        return full_relation

    def output_size_for_input_size(self, input_size, output_node_name=None):
        assert (input_size[0] > 0 and input_size[1] > 0)
        all_sizes = self.get_network_activation_sizes(input_size, verbose=0)
        if output_node_name is None: return all_sizes[str(self.output_nodes_list[-1])][:2]
        else: return all_sizes[str(output_node_name)][:2]

    def get_output_node_index(self, output_node_index=None, output_node_name=None):
        assert(output_node_index is None or output_node_name is None)

        if output_node_name is not None:
            output_node_index_list = [i for i, e in enumerate(self.output_nodes_list) if e == output_node_name]
            assert (len(output_node_index_list))
            output_node_index = output_node_index_list[0]
        elif output_node_index is None:
            output_node_index = -1
        if output_node_index < 0: output_node_index = len(self.components_list)+output_node_index
        if not (output_node_index >= 0 and output_node_index <= (len(self.components_list)-1)):
            print('max_layer must be in a valid range [0, number of layers in the network - 1]: ' + str([1, (len(self.components_list)-1)]))
            quit()
        return output_node_index

    def forward(self, x, output_node_index=None, output_node_name=None):
        output_node_index = self.get_output_node_index(output_node_index=output_node_index, output_node_name=output_node_name)

        output_dict = {'input_node': x}
        for i, layer in enumerate(self.components_list[:output_node_index+1]):

            # if self.incoming_nodes_list[i] == ['input_node']:
            if i == 0:
                assert (self.incoming_nodes_list[i] == ['input_node'])
                curr_out = layer.forward(x)
            else:
                input_list = [output_dict[e] for e in self.incoming_nodes_list[i]]
                assert (len(input_list) > 0)
                if len(input_list) > 1: curr_out = layer.forward(input_list)
                else: curr_out = layer.forward(input_list[0])
            output_dict[self.output_nodes_list[i]] = curr_out

        # print('Output name: ' + str(self.output_nodes_list[i]))
        return curr_out






















