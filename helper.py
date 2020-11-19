from __future__ import print_function
from IPython.core.debugger import Pdb
pdb = Pdb()
trace = pdb.set_trace

import os
import gc
import uuid
import psutil
import resource
import math
import copy
from sklearn.manifold import TSNE
from umap import UMAP
from collections.abc import Iterable

import numpy as np 
np.set_printoptions(suppress=True)
import tensorflow as tf
if tf.__version__ == '1.14.0':
    import tensorflow_probability as tfp
    tf_compat_v1 = tf.compat.v1
    tf_compat_v1.enable_resource_variables()
else:
    tf_compat_v1 = tf

import optimization

def make_exp_dir(global_args):
    random_code = str(uuid.uuid4().hex)
    exp_dir = global_args.main_experiments_dir + global_args.experiment_group + '/' + \
              global_args.algorithm_name + '/' + global_args.dataset_name + '/' + random_code + '/'
    print('\n\nEXPERIMENT RESULT DIRECTORY: '+ exp_dir + '\n\n')
    if os.path.exists(exp_dir): 
        print('Experiment directory already exists. Aborting.'); quit()
    else: 
        os.makedirs(exp_dir)
    return exp_dir

def get_model_and_algorithm_optimization_args(algorithm_name):
    from Algorithms import get_algorithm

    if 'VanillaGAN' in algorithm_name:
        from Models.VanillaGAN import Model
    elif 'fGAN' in algorithm_name:
        from Models.fGAN import Model
    elif 'wGAN' in algorithm_name:
        from Models.wGAN_gp import Model

    algorithm_args, optimization_args = get_algorithm(algorithm_name)
    return Model, algorithm_args, optimization_args 

def read_data_text(data_file_path):
    with open(data_file_path, 'r') as file: line_list = file.readlines()
    
    variables_desc_str_list = (line_list[0].strip()[1:-1].split("}, {"))
    variables_desc_dicts = [eval(line_list[0])] #hacky fix for now
    temporal_desc_str_list = eval(line_list[1])

    value_lines = line_list[4:]
    value_lines_split = [line.strip()[1:-1].split("], [") for line in value_lines]
    value_lines_split_value = np.asarray([line[0].split(', ') for line in value_lines_split]).astype(np.float)
    value_lines_split_temporal = np.asarray([line[1].split(', ') for line in value_lines_split]).astype(np.float)

    return variables_desc_str_list, variables_desc_dicts, temporal_desc_str_list, value_lines_split_value, value_lines_split_temporal

def write_data_text(data_file_path, temporal_dict, data_combinations):
    ordered_temporal_keys = sorted(temporal_dict.keys())
    temporal_values = []
    for key in ordered_temporal_keys: temporal_values.append(temporal_dict[key])
    
    values = data_combinations['value']
    
    #another hacky fix
    if isinstance(values, Iterable):
        values = [np.mean([e[0] for e in values])] 
    else:
        values = [values]

    #values = [e['value'] for e in data_combinations]
    values_str = str(values)
    temporal_str = str(temporal_values)
    overall_str = values_str + ', ' + temporal_str + '\n'        

    if not os.path.exists(data_file_path):
        data_combinations_copy = copy.deepcopy(data_combinations)
        data_combinations_copy.pop('value')    #for e in data_combinations_copy: e.pop('value')
        variable_desc_str = str(data_combinations_copy)
        temporal_desc_str = str(ordered_temporal_keys)
        description_str = variable_desc_str + '\n' + temporal_desc_str + '\n' + '***********************************' + '\n\n'
        with open(data_file_path, 'w+') as file: file.write(description_str)
    with open(data_file_path, 'a') as file: file.write(overall_str)


def get_data_loader(global_args):
    #########################  MNIST  #######################################

    if global_args.dataset_name == 'MNIST': 
        from DataLoaders.MNIST.MNISTLoader import DataLoader  
    elif global_args.dataset_name == 'ColorMNIST': 
        from DataLoaders.MNIST.ColorMNISTLoader import DataLoader

    #########################  CIFAR  #######################################

    elif global_args.dataset_name == 'CIFAR10': 
        from DataLoaders.CIFAR.Cifar10Loader import DataLoader

    #################################################################################
    data_loader = DataLoader(batch_size=global_args.algorithm_args.batch_size*len(global_args.list_of_device_names))
    return data_loader

def assign_to_device(device_name, parameter_device_name=None, parameter_op_names=['VarHandleOp', 'VarIsInitializedOp'], verbose=False):
    if parameter_device_name is None: parameter_device_name = device_name
    def _assign(op):        
        node_def = op if isinstance(op, tf_compat_v1.NodeDef) else op.node_def
        if verbose: print('All Device Ops:', node_def.op)
        if node_def.op in parameter_op_names: 
            if verbose: print('Parameter Device Ops:', node_def.op)
            return parameter_device_name
        else: return device_name
    return _assign

def rejection_sampling_tf(reweighting, max_weight):    
    z_sample = tf.random_uniform(tf.shape(reweighting), 0, max_weight, dtype=tf.float32)
    accept_reject_mask = tf.stop_gradient(tf.cast(z_sample < reweighting, tf.float32))
    return accept_reject_mask

def multiply_fractions(curr_relation, update_relation):
    a, b, c, d = curr_relation
    e, f, g, h = update_relation
    # (e/f)*( (a/b) *x +(c/d) )+(g/h) = ((ea)/(fb))*x+ ( (ec)/(fd) + (g/h)  ) = ((ea)/(fb))*x + ((ech+fdg)/(fdh))
    # = ((ea/gcd(ea, fb))/(fb/gcd(ea, fb))) * x + (( (ech+fdg)/gcd(ech+fdg, fdh) )/( fdh/gcd(ech+fdg, fdh) ))
    # f([a,b,c,d], [e,f,g,h]) = [ea, fb, ech+fdg, fdh] = [ea/gcd(ea, fb), fb/gcd(ea, fb), (ech+fdg)/gcd(ech+fdg, fdh), fdh/gcd(ech+fdg, fdh)]
    # return (e*a)/fractions.gcd(e*a, f*b), (f*b)/fractions.gcd(e*a, f*b), (e*c*h+f*d*g)/fractions.gcd(e*c*h+f*d*g, f*d*h), (f*d*h)/fractions.gcd(e*c*h+f*d*g, f*d*h)
    return (e*a), (f*b), (e*c*h+f*d*g), (f*d*h)

def safe_tf_sqrt(x, clip_value=1e-5):
    return tf.sqrt(tf.clip_by_value(x, clip_value, np.inf))

def tf_nn_upsample_tensor(x, upsample_rate=[2, 2]):
    input_shape = []
    for i, e in enumerate(x.get_shape().as_list()):
        if e is not None: 
            input_shape.append(e)
        else:
            input_shape.append(tf.shape(x)[i])
    
    intermediate_shape = [input_shape[0], input_shape[1]*upsample_rate[0], input_shape[2], input_shape[3]]
    output_shape = [input_shape[0], input_shape[1]*upsample_rate[0], input_shape[2]*upsample_rate[1], input_shape[3]]
    
    x = tf.reshape(tf.concat([x[:, :, np.newaxis, :, :]]*upsample_rate[0], axis=2), intermediate_shape)
    return tf.reshape(tf.concat([x[:, :, :, np.newaxis, :]]*upsample_rate[1], axis=3), output_shape)

def get_device_names(gpu_ID_list):

    if len(gpu_ID_list) > 0: 
        parameter_device_name = '/gpu:'+gpu_ID_list[0]
        list_of_device_names = ['/gpu:'+str(e) for e in gpu_ID_list]
        list_physical_devices = tf.config.experimental.list_physical_devices('GPU')
        for curr_device_name in list_of_device_names:
            try: 
                assert (any(['/physical_device:'+curr_device_name[1:] == e.name.lower() for e in list_physical_devices]))
            except: 
                print('\n\n' + curr_device_name + ' does not match any physical devices detected.'); 
                print('List of physical devices available: '+str(list_physical_devices)+ '\n\n')
                quit()
    else: 
        parameter_device_name = '/cpu:0'
        list_of_device_names = [parameter_device_name]


    return parameter_device_name, list_of_device_names

def filter_tag_vars(tag_list):
    all_vars = tf_compat_v1.trainable_variables()

    filtered_var_list = []
    for var in all_vars:
        # assert (var.name.count('/') == 1)

        var_tag_list = var.name[:var.name.find('/')].split('--')
        tag_found = False
        if len(tag_list) == 0:
            tag_found = True
        else:
            for tag in tag_list:
                if tag in var_tag_list: 
                    tag_found = True
                    break
        if tag_found: filtered_var_list.append(var)

    return filtered_var_list

def save_checkpoint(saver, sess, global_args, global_step, checkpoint_exp_dir, save_meta_graph=True):
    if not os.path.exists(checkpoint_exp_dir+'checkpoint/'): os.makedirs(checkpoint_exp_dir+'checkpoint/')
    saver.save(sess, checkpoint_exp_dir+'checkpoint/model', global_step=global_step)
    if save_meta_graph: 
        saver.export_meta_graph(checkpoint_exp_dir+'checkpoint/model.meta')
    save_specs_files(checkpoint_exp_dir, global_args) # rewrite the Spec.txt

def load_checkpoint(saver, sess, checkpoint_exp_dir, load_meta_graph=True):
    if saver is None or load_meta_graph: 
        saver = tf.train.import_meta_graph(checkpoint_exp_dir+'checkpoint/model.meta')
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_exp_dir+'checkpoint/', latest_filename=None))
    checkpoint_epoch, checkpoint_global_args = load_specs_files(checkpoint_exp_dir)
    return checkpoint_epoch, checkpoint_global_args

def save_specs_files(checkpoint_exp_dir, global_args):
    global_args_copy = copy.deepcopy(global_args)

    #will obviously need to improve this
    del global_args_copy.algorithm_args
    
    args_str = repr(global_args_copy)
    f = open(checkpoint_exp_dir + 'specs.txt', 'w')
    f.write(args_str)
    f.close()

def load_specs_files(checkpoint_exp_dir):
    from argparse import Namespace
    f = open(checkpoint_exp_dir + 'specs.txt', 'r')
    s = f.read()
    f.close()
    checkpoint_global_args = eval(s)
    return checkpoint_global_args.temporal_dict['epoch'], checkpoint_global_args

def tf_safe_log(x, smoothing_param=1e-8, mode='Clip'):
    assert (mode == 'Clip' or mode == 'Add')
    if mode == 'Clip':
        return tf.math.log(tf.clip_by_value(x, smoothing_param, np.inf))
    elif mode == 'Add':
        return tf.math.log(x+smoothing_param)

def Logit(x):
    # return tf.math.log(1e-7+x)-tf.math.log(1e-7+1-x)
    return tf.log(1e-7+x)-tf.log(1e-7+1-x)

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def tf_average_n(list_of_tensors):
    return tf_compat_v1.add_n(list_of_tensors)/float(len(list_of_tensors))

def compute_embeddings(fit_tensor_list, n_fit_samples, mode='UMAP', n_components=2, verbose=True):
    assert (mode == 'UMAP' or mode == 'T-SNE')
    assert (n_components > 1)

    if verbose: print('Preparing the data to be embedded.')
    permuted_indeces = [np.random.permutation(np.arange(e.shape[0])) for e in fit_tensor_list]
    chosen_indeces = [e[:n_fit_samples] for e in permuted_indeces]
    permuted_indeces = None
    gc.collect(); gc.collect()
    chosen_samples = [e[chosen_indeces[i]] for i, e in enumerate(fit_tensor_list)]
    chosen_samples_flat = [e.reshape([e.shape[0], -1]) for e in chosen_samples]
    all_chosen_samples = np.concatenate(chosen_samples_flat, axis=0)
    
    all_permuted_indeces = np.random.permutation(np.arange(all_chosen_samples.shape[0]))
    inverse_all_permuted_indeces = np.zeros((all_chosen_samples.shape[0],), dtype=int)
    for ind, e in enumerate(all_permuted_indeces): inverse_all_permuted_indeces[e] = ind
    all_chosen_samples_permuted = all_chosen_samples[all_permuted_indeces, :]

    if verbose: print('Embedding the data using method: '+mode)
    if mode == 'UMAP':
        all_outputs_permuted = UMAP(n_components=n_components).fit_transform(all_chosen_samples_permuted)
    elif mode == 'T-SNE':
        all_outputs_permuted = TSNE(n_components=n_components).fit_transform(all_chosen_samples_permuted)
    
    if verbose: print('Normalizing the embedding results.')
    all_outputs = all_outputs_permuted[inverse_all_permuted_indeces, :]
    all_outputs_centered = all_outputs-np.mean(all_outputs, axis=0)[np.newaxis, :]
    all_outputs_normalized = all_outputs_centered/(np.std(all_outputs_centered, axis=0)[np.newaxis, :]+1e-7)

    outputs = []
    start_ind = 0
    for i, e in enumerate(chosen_samples_flat):
        outputs.append(all_outputs_normalized[start_ind:start_ind+e.shape[0], :])
        start_ind = start_ind+e.shape[0]

    return outputs, chosen_indeces

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class AccumAttrDict():
    def __init__(self, acc_key_list=[]):
        self.values = AttrDict()
        self.weighting = None
        self.weight = AttrDict()
        self.acc_key_list = acc_key_list

    def add_acc_keys(self, acc_key_list):
        for e in acc_key_list:
            if e not in self.acc_key_list: 
                self.acc_key_list.append(e)

    def add_values(self, add_dict, weight_size_spec=None):
        if self.weighting is None:
            self.weighting = (not (weight_size_spec is None))
        else:
            if (not self.weighting): assert (weight_size_spec is None)
        if self.weighting: 
            assert (type(weight_size_spec) == int or type(weight_size_spec) == dict)
            if type(weight_size_spec) == int: 
                assert (weight_size_spec > 0)
            if type(weight_size_spec) == dict: 
                for key in add_dict: assert (key in weight_size_spec)

        for key in add_dict:
            if key in self.acc_key_list and add_dict[key] is not None:
                if key not in self.values:
                    self.values[key] = None
                    self.weight[key] = None

                if self.values[key] is None:
                    if self.weighting:
                        if type(weight_size_spec) == int:
                            self.values[key] = weight_size_spec*add_dict[key]
                            self.weight[key] = weight_size_spec
                        else: 
                            self.values[key] = weight_size_spec[key]*add_dict[key]
                            self.weight[key] = weight_size_spec[key]
                    else: 
                        self.values[key] = 1*add_dict[key]
                        self.weight[key] = 1
                else:
                    assert (type(self.values[key]) == type(add_dict[key])) or (isinstance(self.values[key], np.floating) and isinstance(add_dict[key], np.floating))
                    if self.weighting:
                        if type(weight_size_spec) == int: 
                            self.values[key] += weight_size_spec*add_dict[key]
                            self.weight[key] += weight_size_spec
                        else: 
                            self.values[key] += weight_size_spec[key]*add_dict[key]
                            self.weight[key] += weight_size_spec[key]
                    else:
                        self.values[key] += 1*add_dict[key]
                        self.weight[key] += 1

    def normalize(self):
        normalized_dict = AttrDict()
        for key in self.values: 
            normalized_dict[key] = float(self.values[key])/float(self.weight[key])
        return normalized_dict

def get_kth_dim_slice_np(dim, start_n, stop_n=None, step_n=None):
    return (slice(None),)*dim+(slice(start_n, stop_n, step_n),)

def interleave_tensors(list_of_tensors, interleave_dim=0):
    assert (len(list_of_tensors) > 0)
    assert (all([e.shape == list_of_tensors[0].shape for e in list_of_tensors]))
    tensor_size = list_of_tensors[0].shape
    interleaved_tensor_size = list(tensor_size[:interleave_dim])+[tensor_size[interleave_dim]*len(list_of_tensors)]+list(tensor_size[interleave_dim+1:])
    interleaved_tensor = np.zeros(interleaved_tensor_size)
    for i in range(len(list_of_tensors)): 
        interleaved_tensor[get_kth_dim_slice_np(interleave_dim, start_n=i, stop_n=None, step_n=len(list_of_tensors))] = list_of_tensors[i]
    return interleaved_tensor

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "{:.3f}".format(s)+" "+str(size_name[i])

def report_instance_memory(): # GB
    mem = psutil.virtual_memory()
    return "{:.3f}".format(mem.used/1024/1024/1024)+' GB/'+ "{:.3f}".format(mem.total/1024/1024/1024)+' GB'

def report_process_memory(): # GB
    return "{:.3f}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/ 1024)+' GB'

def report_tensorflow_memory(sess, memory_node): # GB
    return convert_size(sess.run(memory_node))

def report_memory(sess, memory_node):
    if memory_node is not None:
        return ' | Overall: '+report_instance_memory()+' | Process: '+report_process_memory()+' | TF: '+report_tensorflow_memory(sess, memory_node)
    else:
        return ' | Overall: '+report_instance_memory()+' | Process: '+report_process_memory()

def set_grads_and_vars_for_device(grads_and_vars_all_groups_all_devs, optimized_vars, device_name, computations, optimizers_dict, global_step, optimization_args):
    for group in computations[device_name].objective_dict:
        if group not in optimized_vars: 
            optimized_vars[group] = filter_tag_vars([group])
        if group not in grads_and_vars_all_groups_all_devs: grads_and_vars_all_groups_all_devs[group] = {}
        grads_and_vars_all_groups_all_devs[group][device_name]= optimization.clipped_optimizer_minimize(
            mode=optimization_args[group].optimizer_class, optimizer=optimizers_dict[group], loss=computations[device_name].objective_dict[group], 
            var_list=optimized_vars[group], global_step=global_step, clip_param=optimization_args[group].gradient_clipping)

def sess_run_tf(session, input, feed_dict=None):
    """Wrapper for making Session.run() more user friendly.

    With this function, input can be either a list or a dictionary.

    If input is a list, this function will behave like
    tf.session.run() and return a list in the same order as well. If
    input is a dict then this function will also return a dict where
    the returned values are associated with the corresponding keys from
    the input dict.

    Keyword arguments:
    session -- An open TensorFlow session.
    input -- A list or dict of ops to fetch.
    feed_dict -- The dict of values to feed to the computation graph.
    """
    if hasattr(input, 'keys') and callable(getattr(input, 'keys')) and \
       hasattr(input, 'values') and callable(getattr(input, 'values')):
        keys, values = input.keys(), list(input.values())
        if feed_dict is None:
            res = session.run(values)
        else:
            res = session.run(values, feed_dict)            
        return {key: value for key, value in zip(keys, res)}
    else:
        if feed_dict is None:
            return session.run(input)
        else:
            return session.run(input, feed_dict)


def truncated_gaussian_sampler_np(shape, mu=0, std=1, lower=None, upper=None):
    if upper is None: upper = 2*std
    if lower is None: lower = -2*std

    X = np.zeros(shape, dtype=np.float32)
    X_flattened = np.reshape(X, X.size)
    
    i = 0
    while i < len(X_flattened):
        x = np.random.normal(loc=mu, scale=std)
        if lower <= x - mu <= upper:
            X_flattened[i] = x
            i += 1

    X = np.reshape(X_flattened, shape)
    return X

def tf_replicate_batch_dict(batch_np_example, device_names, batch_size):
    batch_tf = {'all_devices': {}, 'per_device': {}}
    for key in batch_np_example:
        curr_data_size = list(batch_np_example[key].shape[1:])
        if key not in batch_tf['all_devices']: 
            batch_tf['all_devices'][key] = tf_compat_v1.placeholder(tf.float32, [None]+curr_data_size)
    
    for i, device_name in enumerate(device_names):
        batch_tf['per_device'][device_name] = {}
        for key in batch_np_example:
            if len(device_names) > 1:
                batch_tf['per_device'][device_name][key] = batch_tf['all_devices'][key][i*batch_size:(i+1)*batch_size, ...]
            else:
                batch_tf['per_device'][device_name][key] = batch_tf['all_devices'][key]

    def feed_dict_func(batch_np):
        feed_dict = {}
        for key in batch_tf['all_devices']:
            feed_dict[batch_tf['all_devices'][key]] = batch_np[key]
        return feed_dict

    return batch_tf, feed_dict_func

# def merge_dicts(dict_list, deep=True):
#     merged_dict = {}
#     for dicti in (dict_list):
#         for key in dicti.keys():
#             merged_dict[key] = dicti[key]
#     return merged_dict

#     # if deep:
#     #     z = copy.deepcopy(x)   # start with x's keys and values
#     # else:        
#     #     z = x.copy()   # start with x's keys and values
#     # z.update(y)    # modifies z with y's keys and values & returns None
#     # return z

def get_residual_block(block_nm, num_channels, input_node_nm, mode='TRANSPOSE'):
    assert (mode == 'TRANSPOSE' or mode == 'PADDING')
    
    if mode == 'TRANSPOSE':
        return [
            ([input_node_nm], block_nm + '_layer_1', 'Convolution', {'n_out_channels': num_channels, 'kernel_shape': [3, 3], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
            ([block_nm + '_layer_1'], block_nm + '_layer_2', 'BatchNorm', {'mode': 'Regular'}),
            ([block_nm + '_layer_2'], block_nm + '_layer_3', 'ElementwiseApply', {'func': tf.nn.relu}),
            ([block_nm + '_layer_3'], block_nm + '_layer_4', 'TransposedConvolution', {'n_out_channels': num_channels, 'kernel_shape': [3, 3], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': False, 'initializer_mode': 'gaussian', 'force_no_matmul': False}),
            ([block_nm + '_layer_4'], block_nm + '_layer_5', 'BatchNorm', {'mode': 'Regular'}),
            ([input_node_nm, block_nm + '_layer_5'], block_nm + '_layer_6', 'Reduce', {'mode': 'Sum'}),
            ([block_nm + '_layer_6'], block_nm + '_layer_7', 'ElementwiseApply', {'func': tf.nn.relu}),        
        ]
    else:
        return [
            ([input_node_nm], block_nm + '_layer_1', 'Convolution', {'n_out_channels': num_channels, 'kernel_shape': [3, 3], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False, 'padding_mode': 'SAME'}),
            ([block_nm + '_layer_1'], block_nm + '_layer_2', 'BatchNorm', {'mode': 'Regular'}),
            ([block_nm + '_layer_2'], block_nm + '_layer_3', 'ElementwiseApply', {'func': tf.nn.relu}),
            ([block_nm + '_layer_3'], block_nm + '_layer_4', 'Convolution', {'n_out_channels': num_channels, 'kernel_shape': [3, 3], 'strides': [1, 1], 'dilations': [1, 1], 'use_bias': True, 'initializer_mode': 'gaussian', 'force_no_matmul': False, 'padding_mode': 'SAME'}),
            ([block_nm + '_layer_4'], block_nm + '_layer_5', 'BatchNorm', {'mode': 'Regular'}),
            ([input_node_nm, block_nm + '_layer_5'], block_nm + '_layer_6', 'Reduce', {'mode': 'Sum'}),
            ([block_nm + '_layer_6'], block_nm + '_layer_7', 'ElementwiseApply', {'func': tf.nn.relu}),        
        ]



