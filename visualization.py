#contains all helper functions used in visualization

from __future__ import print_function
from IPython.core.debugger import Pdb
pdb = Pdb()
trace = pdb.set_trace

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
plt.rcParams['font.family'] = 'serif' 
# if this is not found $ sudo apt install msttcorefonts -qq  
# tab+enter to accept $ rm ~/.cache/matplotlib -rf
plt.rcParams['font.serif'] = 'Times New Roman'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 16
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['figure.titlesize'] = 12

import os
import glob
import re
import scipy
from PIL import Image
import copy

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

################################################################################################################################################
#grabs tensors from all devices and concatenates them
def aggregate_model_tensor_across_devices(tf_graph, on_device, model, variable_name):
    with tf_graph.as_default():
        with tf.device(on_device):
            if len(model.computations[list(model.computations.keys())[0]][variable_name].get_shape()) == 0:
                return tf.concat([model.computations[curr_device_name][variable_name][np.newaxis] for curr_device_name in model.computations], axis=0)
            else:
                return tf.concat([model.computations[curr_device_name][variable_name] for curr_device_name in model.computations], axis=0)    

#takes the provided visualization_node and obtains the corresponding tensors, performs the required computations in feed_data_func, and returns results as np arrays
def compute_variable_for_visualization(sess, visualization_node, data_loader, feed_dict_func, unique_data_combination, verbose=False):
    if not ('randomized' in unique_data_combination): data_loader.setup(stage=unique_data_combination['args']['data_mode'], verbose=False)
    else: data_loader.setup(stage=unique_data_combination['data_mode'], randomized=unique_data_combination['randomized'], verbose=False)
    
    if verbose:
        print('\n\nComputing the following data: ')
        print(unique_data_combination)
        print('\n\n')

    n_seen_samples, n_used_samples = 0, 0
    scalar_flag = (len(visualization_node.shape) == 0) or ((len(visualization_node.shape) == 1) and (visualization_node.get_shape()[0].value == 1))
    if scalar_flag: acc_result = helper.AccumAttrDict(['visualization_node'])
    else: acc_result = None
    
    if type(unique_data_combination['args']['sample_spec']) == tuple:
        assert (not ('randomized' in unique_data_combination))
        chosen_sample_indices = np.asarray(unique_data_combination['args']['sample_spec'])

    for _, curr_batch_size, batch_np in data_loader:
        if curr_batch_size == 0: trace()

        if unique_data_combination['args']['sample_spec'] == 'All':
            effective_batch_size = curr_batch_size

        elif type(unique_data_combination['args']['sample_spec']) == int:
            effective_batch_size = curr_batch_size-max(0, n_used_samples+curr_batch_size-unique_data_combination['args']['sample_spec'])
            if effective_batch_size == 0: break

            # print('visualization effective_batch_size, sample_spec:') 
            # print(effective_batch_size, unique_data_combination['sample_spec'])
            for e in batch_np: 
                if effective_batch_size < batch_np[e].shape[0]: 
                    batch_np[e] = batch_np[e][:effective_batch_size, ...]

        elif type(unique_data_combination['args']['sample_spec']) == tuple:
            start_index = n_seen_samples
            end_index = n_seen_samples+curr_batch_size
            if start_index > chosen_sample_indices[-1]: break

            curr_batch_mask = (chosen_sample_indices >= start_index)*(chosen_sample_indices < end_index) 
            curr_batch_ind = chosen_sample_indices[curr_batch_mask]
            effective_batch_size = len(curr_batch_ind)
            if effective_batch_size == 0: continue

            # print('visualization effective_batch_size, len(sample_spec):') 
            # print(effective_batch_size, len(unique_data_combination['sample_spec']))
            curr_batch_ind_relative = curr_batch_ind-start_index
            for e in batch_np: batch_np[e] = batch_np[e][curr_batch_ind_relative, ...]
            
        curr_result_dict = sess.run({'visualization_node': visualization_node}, feed_dict=feed_dict_func(batch_np))
        if scalar_flag:
            # if effective_batch_size < 1: trace()
            acc_result.add_values(curr_result_dict, weight_size_spec=effective_batch_size)
        else:
            if acc_result is None: acc_result = curr_result_dict['visualization_node']
            else: acc_result = np.concatenate([acc_result, curr_result_dict['visualization_node']], axis=0)

        n_used_samples += effective_batch_size
        n_seen_samples += curr_batch_size

    if scalar_flag:
        acc_result = acc_result.normalize()['visualization_node']
    return acc_result

################################################################################################################################################
#takes in the visualizations specified in the model definition and produces them
def visualize_model(sess, model, data_loader, feed_dict_func, tf_graph, on_device, temporal_dict, exp_dir):
    variable_names_set = set()
    for vis_group_name, vis_group_info in model.visualizations.items():
        variable_names_set.update(vis_group_info['variables']) 

    variable_names_set.remove('inception_score') #more hackiness
    visualization_nodes = {}
    for variable_name in variable_names_set:
        visualization_nodes[variable_name] = aggregate_model_tensor_across_devices(tf_graph, on_device, model, variable_name)

    pipeline_dict = {
        'Image Matrix': image_matrix_pipeline,
        'Class Image Matrix': class_image_matrix_pipeline,
        'Line Plot': line_plot_pipeline,
        'Embedding': embedding_pipeline,
        'Histogram': histogram_pipeline,
        'Store': store_pipeline
    }

    #go through all the visualizations in the vis dict
    for vis_group_name, vis_group_info in model.visualizations.items():
        vis_pipeline_func = pipeline_dict[vis_group_info['pipeline']]
        vis_group_info['name'] = vis_group_name

        if vis_group_name == 'Inception Score Generated Samples': 
            from InceptionScore import inception_score
            vis_group_info['value'] = inception_score(model.visualizations['Generative Model Sample']['value'])[0] #get the value of the metric that needs to be visualized
        
        else:
            visualization_node = visualization_nodes[vis_group_info['variables'][0]] 
            vis_group_info['value'] = compute_variable_for_visualization(sess, visualization_node, data_loader, feed_dict_func, vis_group_info) #get the value of the metric that needs to be visualized

        vis_pipeline_func(vis_group_info, temporal_dict, exp_dir) #execute the corresponding visualization func using the newly computed values

################################################################################################################################################
#visualization of image samples
def image_matrix_pipeline(group, temporal_dict, exp_dir):
    visualization_name = group['name']
    visualization_arguments = group['args']
    tensor_list = group['value']
    file_path = exp_dir + 'Visualizations/' + (visualization_name.lower().replace(' ', '_'))

    if 'agg_mode' in visualization_arguments: file_path = file_path + '_' + visualization_arguments['agg_mode']
    if 'save_temporally' in visualization_arguments:
        temporal_dir = file_path + '/'
        if not os.path.exists(temporal_dir): os.makedirs(temporal_dir)
        temporal_file_path = temporal_dir + str(temporal_dict[visualization_arguments['save_temporally']]) + '_' + \
            (visualization_arguments['save_temporally'].lower().replace(' ', '_')) + '_' + (visualization_name.lower().replace(' ', '_'))
        make_image_matrix(tensor_list, [file_path, temporal_file_path], **visualization_arguments)
    else:
        make_image_matrix(tensor_list, [file_path]) #**visualization_arguments)

#line plot visualizations
def line_plot_pipeline(group, temporal_dict, exp_dir):
    store_pipeline(group, temporal_dict, exp_dir)

    visualization_name = group['name']
    visualization_arguments = group['args']

    data_file_path = exp_dir + 'ResultData/' + (visualization_name.lower().replace(' ', '_')) + '.txt'
    variables_desc_str_list, value_desc_dicts, temporal_name_list, values, temporal = helper.read_data_text(data_file_path)

    line_plot_path = exp_dir + 'Visualizations/' + (visualization_name.lower().replace(' ', '_')) + '.png'
    make_lineplot(values, temporal, value_desc_dicts, temporal_name_list, visualization_arguments['x_axis'], save_path_list=[line_plot_path])

#storing experiment data in .txt files
def store_pipeline(group, temporal_dict, exp_dir):
    visualization_name = group['name']
    visualization_arguments = group['args']
    
    data_file_path = exp_dir + 'ResultData/' + (visualization_name.lower().replace(' ', '_')) + '.txt'
    if not os.path.exists(exp_dir + 'ResultData/'): os.makedirs(exp_dir + 'ResultData/')
    helper.write_data_text(data_file_path, temporal_dict, group)
    group['group_element'] = True

################################################################################################################################################

#TODO: figure out a way ard the hacky fixes here
def make_image_matrix(tensor_list, save_path_list=[], agg_mode='Top-Bottom', block_size=None, max_rows=None, padding=[4, 4], save_temporally=None):
    assert (agg_mode == 'Interlaced' or agg_mode == 'Top-Bottom')
    assert (len(tensor_list) > 0)
    
    tensor_list = [np.expand_dims(e, axis=0) for e in tensor_list if len(e.shape) == 3] #hacky fix
    for e in tensor_list: assert (len(e.shape) == 4)

    if len(tensor_list) == 1:
        tensor_np = tensor_list[0]
    else:
        if agg_mode == 'Interlaced': 
            tensor_np = helper.interleave_tensors(tensor_list, interleave_dim=0)
        elif agg_mode == 'Top-Bottom':
            tensor_np = np.concatenate(tensor_list, axis=0)
    
    visualize_image_matrix(tensor_np, block_size=block_size, max_rows=max_rows, padding=padding, save_path_list=save_path_list)

def make_lineplot(values, temporal, value_desc_dicts, temporal_name_list, x_axis_name, save_path_list):
    temporal_index = [i for i, e in enumerate(temporal_name_list) if e == x_axis_name][0]
    x_axis = temporal[:, temporal_index, np.newaxis]
    variable_list = [e['variables'] for e in value_desc_dicts]
    visualize_lineplot(values, x_axis, line_names=variable_list, x_axis_name=x_axis_name, save_path_list=save_path_list)

################################################################################################################################################

def visualize_image_matrix(tensor_np, block_size=None, max_rows=None, padding=[4, 4], save_path_list=None):
    assert (len(tensor_np.shape) == 4 or len(tensor_np.shape) == 5)
    assert (tensor_np.shape[-1] == 1 or tensor_np.shape[-1] == 3)
    image_size = list(tensor_np.shape[-3:])

    if block_size is not None:
        if len(tensor_np.shape) == 4:
            assert(tensor_np.shape[0] == np.prod(block_size))
            tensor_np = tensor_np.reshape(block_size+image_size)
        elif len(tensor_np.shape) == 5:
            assert (np.prod(tensor_np.shape[:2]) == np.prod(block_size))
            if tensor_np.shape[:2] != block_size:
                tensor_np = tensor_np.reshape(block_size+image_size)
    else:
        if len(tensor_np.shape) == 4:
            batch_size_sqrt_floor = int(np.floor(np.sqrt(tensor_np.shape[0])))
            block_size = [batch_size_sqrt_floor, batch_size_sqrt_floor]
            tensor_np = tensor_np[:np.prod(block_size), ...].reshape(block_size+image_size)
        elif len(tensor_np.shape) == 5:
            block_size = tensor_np.shape[:2]

    if max_rows is None: max_rows = tensor_np.shape[0]    
    canvas = np.ones([image_size[0]*min(block_size[0], max_rows)+ padding[0]*(min(block_size[0], max_rows)+1), 
                      image_size[1]*block_size[1]+ padding[1]*(block_size[1]+1), image_size[2]])
    for i in range(min(block_size[0], max_rows)):
        start_coor = padding[0] + i*(image_size[0]+padding[0])
        for t in range(block_size[1]):
            y_start = (t+1)*padding[1]+t*image_size[1]
            canvas[start_coor:start_coor+image_size[0], y_start:y_start+image_size[1], :] =  tensor_np[i][t]
    if canvas.shape[2] == 1: canvas = np.repeat(canvas, 3, axis=2)

    full_save_path_list = []
    if save_path_list is None:
        scipy.misc.imshow(canvas)
    else:
        for path in save_path_list:
            path_dir = path[:-((path[::-1]).find('/'))]
            if not os.path.exists(path_dir): os.makedirs(path_dir)
            full_path = path+'_ImageMatrix.png'
            Image.fromarray((canvas * 255).astype(np.uint8)).save(full_path) #Deprecated: scipy.misc.toimage(canvas).save(full_path)
            full_save_path_list.append(full_path)
    return full_save_path_list

def visualize_lineplot(vals, times, line_names=None, x_axis_name=None, y_axis_name=None, colors=None, dpi=500, width_height_factors=[9, 6], save_path_list=None):
    assert (1 <= len(vals.shape) == len(times.shape) <= 2)
    if len(vals.shape) == 1:
        vals = vals[np.newaxis, :]
        times = times[np.newaxis, :]

    assert (vals.shape == times.shape) #assert (vals.shape[1] == times.shape[1])
    assert (line_names is None or len(line_names) == vals.shape[1])
    assert (colors is None or len(colors) == vals.shape[1])

    plt.figure(figsize=(width_height_factors[0], width_height_factors[1]), dpi=dpi)
    plt.cla()

    max_x_val = 0
    min_y_val = np.min(vals)
    max_y_val = np.max(vals)
    y_range = max_y_val - min_y_val

    if colors is None:
        if vals.shape[1] <= 11:
            colors = ['r', 'g', 'b', 'k', 'c', 'm', 'gold', 'teal', 'springgreen', 'lightcoral', 'darkgray']
        else:
            HSV_tuples = [(x*1.0/vals.shape[1], 0.5, 0.5) for x in range(N)]
            colors = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

    for i in range(vals.shape[1]):
        val = vals[:,i]
        if times.shape[1] == 1: time = times[:,0]
        else: time = times[:,i]
        plt.plot(time, val, color=colors[i], label=line_names[i], markersize=10)

    if x_axis_name is not None: plt.xlabel(x_axis_name, fontsize=16)
    if y_axis_name is not None: plt.ylabel(y_axis_name, fontsize=16)

    plt.grid()
    plt.legend(frameon=True)
    if (min_y_val-0.1*y_range) != (max_y_val+0.1*y_range):
        plt.ylim((min_y_val-0.1*y_range, max_y_val+0.1*y_range))
    if 0 != (np.max(times)):
        plt.xlim((0, np.max(times)))
    
    full_save_path_list = []
    if save_path_list is None:
        plt.show()
    else:
        for path in save_path_list:
            path_dir = path[:-((path[::-1]).find('/'))]
            if not os.path.exists(path_dir): os.makedirs(path_dir)
            
            full_path = path[:-4] +'_LinePlot.png' #chop off that '.png'
            plt.savefig(full_path, dpi=dpi, facecolor='w', edgecolor='w', orientation='portrait', papertype=None,
                format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)
            full_save_path_list.append(full_path)
    return full_save_path_list

################################################################################################################################################
