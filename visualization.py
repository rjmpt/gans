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

def aggregate_model_tensor_across_devices(tf_graph, on_device, model, variable_name):
    with tf_graph.as_default():
        with tf.device(on_device):
            if len(model.computations[list(model.computations.keys())[0]][variable_name].get_shape()) == 0:
                return tf.concat([model.computations[curr_device_name][variable_name][np.newaxis] for curr_device_name in model.computations], axis=0)
            else:
                return tf.concat([model.computations[curr_device_name][variable_name] for curr_device_name in model.computations], axis=0)    


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

'''
def visualize_model_old(sess, model, data_loader, feed_dict_func, tf_graph, on_device, temporal_dict, exp_dir):
    visualization_nodes = {}

    variable_names_list = []
    for vis_group in model.visualizations.keys():
        curr_var_list = model.visualizations[vis_group]['variables']
        variable_names_list += curr_var_list

        for e in model.visualizations[vis_group]:#['group']: 
            #curr_var_list = [data_combination['variable'] for data_combination in e['data_combinations']]            
            variable_names_list += curr_var_list
        
    variable_names_list = list(set(variable_names_list))

    for variable_name in variable_names_list:
        visualization_nodes[variable_name] = aggregate_model_tensor_across_devices(tf_graph, on_device, model, variable_name)

    for vis_group in model.visualizations.keys():
        if model.visualizations[vis_group]['pipeline']

    for i in range(len(model.visualizations)):
        if model.visualizations[i]['mode'] == 'Image Matrix':
            vis_pipeline_func = image_matrix_pipeline
        elif model.visualizations[i]['mode'] == 'Class Image Matrix':
            vis_pipeline_func = class_image_matrix_pipeline
        elif model.visualizations[i]['mode'] == 'Line Plot':
            vis_pipeline_func = line_plot_pipeline           
        elif model.visualizations[i]['mode'] == 'Embedding':
            vis_pipeline_func = embedding_pipeline
        elif model.visualizations[i]['mode'] == 'Histogram':
            vis_pipeline_func = histogram_pipeline
        elif model.visualizations[i]['mode'] == 'Store':
            vis_pipeline_func = store_pipeline
        
        group_copy = copy.deepcopy(model.visualizations[i]['group'])
        data_combinations = []
        for e in group_copy: data_combinations += e['data_combinations']
        unique_data_combinations = [dict(y) for y in set(tuple(x.items()) for x in data_combinations)]

        for unique_data_combination in unique_data_combinations:
            visualization_node = visualization_nodes[unique_data_combination['variable']]
            unique_data_combination['value'] = compute_variable_for_visualization(sess, visualization_node, data_loader, feed_dict_func, unique_data_combination)

        for data_combination in data_combinations:
            for i, unique_data_combination in enumerate(unique_data_combinations):
                found = True
                for e in data_combination:
                    if (e not in unique_data_combination) or (data_combination[e] != unique_data_combination[e]):
                        found = False; break
                if found == True: 
                    data_combination['value'] = unique_data_combination['value']; break

        vis_pipeline_func(group_copy, temporal_dict, exp_dir)
    '''
################################################################################################################################################
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

    for vis_group_name, vis_group_info in model.visualizations.items():
        vis_pipeline_func = pipeline_dict[vis_group_info['pipeline']]
        vis_group_info['name'] = vis_group_name
 
        #super hacky fix here
        if vis_group_name == 'Inception Score Generated Samples': 
            from InceptionScore import inception_score
            vis_group_info['value'] = inception_score(model.visualizations['Generative Model Sample']['value'])[0]
        
        else:
            visualization_node = visualization_nodes[vis_group_info['variables'][0]] #hacky hardcoding for now
            vis_group_info['value'] = compute_variable_for_visualization(sess, visualization_node, data_loader, feed_dict_func, vis_group_info)

        vis_pipeline_func(vis_group_info, temporal_dict, exp_dir)

        '''
        group_copy = copy.deepcopy(model.visualizations[i]['group'])
        data_combinations = []
        for e in group_copy: data_combinations += e['data_combinations']
        unique_data_combinations = [dict(y) for y in set(tuple(x.items()) for x in data_combinations)]

        for unique_data_combination in unique_data_combinations:
            visualization_node = visualization_nodes[unique_data_combination['variables']]
            unique_data_combination['value'] = compute_variable_for_visualization(sess, visualization_node, data_loader, feed_dict_func, unique_data_combination)

        for data_combination in data_combinations:
            for i, unique_data_combination in enumerate(unique_data_combinations):
                found = True
                for e in data_combination:
                    if (e not in unique_data_combination) or (data_combination[e] != unique_data_combination[e]):
                        found = False; break
                if found == True: 
                    data_combination['value'] = unique_data_combination['value']; break

        vis_pipeline_func(group_copy, temporal_dict, exp_dir)
        '''

################################################################################################################################################

#TODO: fix what this visualization_arguments cheese is doing
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


def class_image_matrix_pipeline(group, temporal_dict, exp_dir):
    for group_element in group:
        visualization_name = group_element['visualization_name']
        visualization_arguments = group_element['args']
        data_combinations = group_element['data_combinations']
        tensor_list = [e['value'] for e in data_combinations]

        image_samples = tensor_list[0]
        image_class_masks = tensor_list[1]
        class_indeces = []
        for i in range(image_class_masks.shape[1]):
            class_indeces.append(np.where(image_class_masks[:, i])[0][:visualization_arguments['n_class_samples']])
        class_image_samples = [image_samples[e, ...] for e in class_indeces]

        file_path = exp_dir + 'Visualizations/' + (visualization_name.lower().replace(' ', '_'))
        if 'mode' in visualization_arguments: file_path = file_path + '_' + visualization_arguments['mode']

        if 'save_temporally' in visualization_arguments:
            temporal_dir = file_path + '/'
            if not os.path.exists(temporal_dir): os.makedirs(temporal_dir)
            temporal_file_path = temporal_dir + str(temporal_dict[visualization_arguments['save_temporally']]) + '_' + \
                (visualization_arguments['save_temporally'].lower().replace(' ', '_')) + '_' + (visualization_name.lower().replace(' ', '_'))
            make_class_image_matrix(class_image_samples, [file_path, temporal_file_path], **visualization_arguments)
        else:
            make_class_image_matrix(class_image_samples, [file_path], **visualization_arguments)


def line_plot_pipeline(group, temporal_dict, exp_dir):
    store_pipeline(group, temporal_dict, exp_dir)

    visualization_name = group['name']
    visualization_arguments = group['args']

    data_file_path = exp_dir + 'ResultData/' + (visualization_name.lower().replace(' ', '_')) + '.txt'
    variables_desc_str_list, value_desc_dicts, temporal_name_list, values, temporal = helper.read_data_text(data_file_path)

    line_plot_path = exp_dir + 'Visualizations/' + (visualization_name.lower().replace(' ', '_')) + '.png'
    make_lineplot(values, temporal, value_desc_dicts, temporal_name_list, visualization_arguments['x_axis'], save_path_list=[line_plot_path])


def embedding_pipeline(group, temporal_dict, exp_dir):
    for group_element in group:
        visualization_name = group_element['visualization_name']
        visualization_arguments = group_element['args']
        data_combinations = group_element['data_combinations']
        tensor_list = [e['value'] for e in data_combinations]

        fit_tensor_list = [tensor_list[i] for i in visualization_arguments['fit_indices']]
        n_fit_samples = min([e.shape[0] for e in fit_tensor_list])
        if 'max_fit_samples' in visualization_arguments:
            n_fit_samples = min(visualization_arguments['max_fit_samples'], n_fit_samples)

        for e in fit_tensor_list: assert (e.shape[1:] == fit_tensor_list[0].shape[1:])
        embeddings, used_indeces = helper.compute_embeddings(fit_tensor_list, n_fit_samples, mode=visualization_arguments['mode'])
        class_mask_list = [None if e is None else tensor_list[e] for e in visualization_arguments['class_pairings']]
        
        file_path = exp_dir + 'Visualizations/' + (visualization_name.lower().replace(' ', '_'))
        if 'save_temporally' in visualization_arguments:
            temporal_dir = file_path + '/'
            if not os.path.exists(temporal_dir): os.makedirs(temporal_dir)
            temporal_file_path = temporal_dir + str(temporal_dict[visualization_arguments['save_temporally']]) + '_' + \
                (visualization_arguments['save_temporally'].lower().replace(' ', '_')) + '_' + (visualization_name.lower().replace(' ', '_'))
            make_embeddings(embeddings, class_mask_list, [file_path, temporal_file_path], title=visualization_name)
        else:
            make_embeddings(embeddings, class_mask_list, [file_path], title=visualization_name)

def histogram_pipeline(group, temporal_dict, exp_dir):
    for group_element in group:
        visualization_name = group_element['visualization_name']
        visualization_arguments = group_element['args']
        data_combinations = group_element['data_combinations']
        tensor_list = [e['value'] for e in data_combinations]
        file_path = exp_dir + 'Visualizations/' + (visualization_name.lower().replace(' ', '_'))
        if 'agg_mode' in visualization_arguments: file_path = file_path + '_' + visualization_arguments['agg_mode']

        if 'save_temporally' in visualization_arguments:
            temporal_dir = file_path + '/'
            if not os.path.exists(temporal_dir): os.makedirs(temporal_dir)
            temporal_file_path = temporal_dir + str(temporal_dict[visualization_arguments['save_temporally']]) + '_' + \
                (visualization_arguments['save_temporally'].lower().replace(' ', '_')) + '_' + (visualization_name.lower().replace(' ', '_'))
            make_histograms(tensor_list, [file_path, temporal_file_path], title=visualization_name, **visualization_arguments)
        else:
            make_histograms(tensor_list, [file_path], title=visualization_name, **visualization_arguments)

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

def make_class_image_matrix(tensor_list, save_path_list=[], mode='Column', n_class_samples=10, padding=[4, 4], save_temporally=None):
    assert (mode == 'Column' or mode == 'Squares')
    for e in tensor_list: assert (len(e.shape) == 4)

    if mode == 'Column': 
        tensor_np = np.concatenate([e[:, np.newaxis, ...] for e in tensor_list], axis=1)
        visualize_image_matrix(tensor_np, max_rows=None, padding=padding, save_path_list=save_path_list)
    elif mode == 'Squares':
        square_edge_num = int(np.sqrt(n_class_samples))
        tensor_np = np.zeros([square_edge_num, square_edge_num*len(tensor_list)] + list(tensor_list[0].shape[1:]))
        for k in range(len(tensor_list)):
            for i in range(square_edge_num):
                for j in range(square_edge_num):
                    tensor_np[i, k*square_edge_num+j, ...] = tensor_list[k][i*square_edge_num+j]
        visualize_image_matrix(tensor_np, max_rows=None, padding=padding, save_path_list=save_path_list)

def make_lineplot(values, temporal, value_desc_dicts, temporal_name_list, x_axis_name, save_path_list):
    temporal_index = [i for i, e in enumerate(temporal_name_list) if e == x_axis_name][0]
    x_axis = temporal[:, temporal_index, np.newaxis]
    variable_list = [e['variables'] for e in value_desc_dicts]
    visualize_lineplot(values, x_axis, line_names=variable_list, x_axis_name=x_axis_name, save_path_list=save_path_list)

def make_embeddings(embeddings, class_mask_list, save_path_list=[], title=''):
    assert (len(embeddings) > 0)
    visualize_embeddings(embeddings, class_mask_list, title=title, save_path_list=save_path_list)

def make_histograms(tensor_list, save_path_list=[], title='', save_temporally=None):
    assert (len(tensor_list) > 0)
    visualize_histograms(tensor_list, title=title, save_path_list=save_path_list)

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

def visualize_histograms(tensor_list, title='', n_bins=None, line_names=None, colors=None, dpi=500, width_height_factors=[9, 6], save_path_list=None):
    flattened_tensor_list = [e.flatten() for e in tensor_list]
    n_histograms = len(flattened_tensor_list)
    if n_bins is None:
        n_bins = [50]*n_histograms
    else:
        assert len(n_bins) == n_histograms
    
    assert (line_names is None or len(line_names) == n_histograms)
    assert (colors is None or len(colors) == n_histograms)
    if colors is None:
        if n_histograms <= 11:
            colors = ['r', 'g', 'b', 'k', 'c', 'm', 'gold', 'teal', 'springgreen', 'lightcoral', 'darkgray']
        else:
            HSV_tuples = [(x*1.0/n_histograms, 0.5, 0.5) for x in range(N)]
            colors = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

    if n_histograms > 1:
        fig, axs = plt.subplots(n_histograms, figsize=(width_height_factors[0], width_height_factors[1]*n_histograms), dpi=dpi)
        fig.suptitle(title, fontsize=21)

    for i in range(n_histograms):
        if n_histograms > 1:
            axs[i].hist(flattened_tensor_list[i], bins=n_bins[i], color=colors[i])
        else:
            plt.hist(flattened_tensor_list[i], bins=n_bins[i], color=colors[i])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    full_save_path_list = []
    if save_path_list is None:
        plt.show()
    else:
        for path in save_path_list:
            path_dir = path[:-((path[::-1]).find('/'))]
            if not os.path.exists(path_dir): os.makedirs(path_dir)
            full_path = path+'_Histogram.png'
            plt.savefig(full_path, dpi=dpi, facecolor='w', edgecolor='w', orientation='portrait', papertype=None,
                format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)
            full_save_path_list.append(full_path)
    return full_save_path_list

def visualize_embeddings(embeddings, class_mask_list, title='', dpi=500, width_height_factors=[4, 4], save_path_list=None):
    n_sets = len(embeddings)

    if n_sets <= 11:
        main_colors = ['r', 'g', 'b', 'k', 'c', 'm', 'gold', 'teal', 'springgreen', 'lightcoral', 'darkgray']
    else:
        HSV_tuples = [(x*1.0/n_sets, 0.5, 0.5) for x in range(N)]
        main_colors = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

    fig, axs = plt.subplots(n_sets+1, figsize=(width_height_factors[0], width_height_factors[1]*(n_sets+1)), dpi=dpi)
    fig.suptitle(title, fontsize=21)

    for i in range(n_sets):
        if class_mask_list[i] is not None:
            n_curr_classes = class_mask_list[i].shape[1]
            if n_curr_classes <= 11:
                class_colors = ['r', 'g', 'b', 'k', 'c', 'm', 'gold', 'teal', 'springgreen', 'lightcoral', 'darkgray']
            else:
                HSV_tuples = [(x*1.0/n_curr_classes, 0.5, 0.5) for x in range(N)]
                class_colors = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
            trace()
        else:
            axs[i].scatter(embeddings[i][:, 0], embeddings[i][:, 1], color=main_colors[i], s=1)

        axs[n_sets].scatter(embeddings[i][:, 0], embeddings[i][:, 1], color=main_colors[i], s=1)

    for i in range(n_sets+1): axs[i].set_aspect('equal')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    full_save_path_list = []
    if save_path_list is None:
        plt.show()
    else:
        for path in save_path_list:
            path_dir = path[:-((path[::-1]).find('/'))]
            if not os.path.exists(path_dir): os.makedirs(path_dir)
            full_path = path+'_Embedding.png'
            plt.savefig(full_path, dpi=dpi, facecolor='w', edgecolor='w', orientation='portrait', papertype=None,
                format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)
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
    # assert (line_widths is None or len(line_widths) == vals.shape[1])
    # assert (line_styles is None or len(line_styles) == vals.shape[1])

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
        # if line_widths: line_width = line_widths[i]
        # if line_styles: line_style = line_styles[i]

        # plt.plot(time, val, linewidth=line_width, linestyle=line_style, color=colors[i], label=line_names[i], markersize=10)
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









































































# def visualization_pipelines(visualizer, tf_graph, sess, data_loader, model, on_device, epoch, exp_dir):
#     for visualization_pipeline in model.visualizations:
#         for visualization_element in model.visualizations[visualization_pipeline]:
#             for variable_name in visualization_element['variables']:
#                 if variable_name not in visualizer:
#                     visualizer[variable_name] = aggregate_model_tensor_across_devices(
#                         tf_graph, on_device, model, variable_name)


#     for visualization_pipeline in model.visualizations:
#         if visualization_pipeline == 'Image Matrix':
#             image_matrix_pipeline(sess, data_loader, visualizer, model.visualizations[visualization_pipeline])
#         elif visualization_pipeline == 'Line Plot':
#             line_plot_pipeline(sess, data_loader, visualizer, model.visualizations[visualization_pipeline])

#     trace()

#     # for visualization_name in model.visualizations:
#     #     curr_variables = [visualizer[variable_name] for variable_name in model.visualizations[visualization_name]['variables']]
#     #     curr_data_settings = model.visualizations[visualization_name]['data_settings']

#     #     if model.visualizations[visualization_name]['pipeline'] == 'Image Matrix':
#     #         image_matrix_pipeline(visualization_name, sess, data_loader, curr_variables, curr_data_settings, **model.visualizations[visualization_name]['args'])
#     #     elif model.visualizations[visualization_name]['pipeline'] == 'Line Plot':
#     #         trace()
#     #     elif model.visualizations[visualization_name]['pipeline'] == 'Distribution Traversal':
#     #         trace()


    
#     # if 'VanillaGAN' in global_args.algorithm_name:
#     #     with tf_graph.as_default():
#     #         aggregate_input_images = tf.concat([model.computations[curr_device_name].input_images for curr_device_name in global_args.list_of_device_names], axis=0)
#     #         aggregate_transformed_sample = tf.concat([model.computations[curr_device_name].transformed_sample for curr_device_name in global_args.list_of_device_names], axis=0)
#     #     compute_dict = {'Input Samples': aggregate_input_images, 'Generated Samples': aggregate_transformed_sample}
#     #     batch_x, batch_y = mnist.train.next_batch(global_args.algorithm_args.batch_size*len(global_args.list_of_device_names))
#     #     curr_feed_dict = {X: batch_x, Y: batch_y}
#     #     results_dict = helper.sess_run_tf(sess, compute_dict, feed_dict=curr_feed_dict)

#     #     vis_dir = global_args.exp_dir+'Visualizations/'
#     #     if not os.path.exists(vis_dir): os.makedirs(vis_dir)
#     #     visualization.visualize_tensor(results_dict['Generated Samples'], save_path_list=[vis_dir + 'GeneratedSamples.png',])
#     #     visualization.visualize_tensor(results_dict['Input Samples'], save_path_list=[vis_dir + 'InputSamples.png',])
#     #     visualization.visualize_tensor(np.concatenate([results_dict['Generated Samples'], results_dict['Input Samples']], axis=0), save_path_list=[vis_dir + 'InputAndGeneratedSamples.png',])
#     #     print('Min max input:', [results_dict['Input Samples'].min(), results_dict['Input Samples'].max()])
#     #     print('Min max generated:', [results_dict['Generated Samples'].min(), results_dict['Generated Samples'].max()])

#     # elif 'SimpleClassifier' in global_args.algorithm_name:
#     #     with tf_graph.as_default():
#     #         mean_accuracy = tf.reduce_mean(tf.concat([model.computations[curr_device_name].accuracy[np.newaxis] for curr_device_name in global_args.list_of_device_names], axis=0))
#     #     test_acc_list = []
#     #     for i in range(0, len(mnist.test.images), len(global_args.list_of_device_names)*global_args.algorithm_args.batch_size):
#     #         curr_feed_dict = {X: mnist.test.images[i:i+len(global_args.list_of_device_names)*global_args.algorithm_args.batch_size],
#     #                           Y: mnist.test.labels[i:i+len(global_args.list_of_device_names)*global_args.algorithm_args.batch_size]}
#     #         test_acc_list.append(sess.run(mean_accuracy, feed_dict=curr_feed_dict))

#     #     print('Accuracy: ' + str(np.mean(test_acc_list)))
    
#     # return vis_tensor_tf_dict









# def line_plot_pipeline(sess, data_loader, visualizer, visualization_elements_list):
#     trace()

# def image_matrix_pipeline(sess, data_loader, visualizer, visualization_elements_list, n_samples=400):

#     # result_dict = {}
#     for data_setting in ['Training', 'Test']:
#         compute_dict = {}
#         for visualization_element in visualization_elements_list:
#             for i, variable_name in enumerate(visualization_element['variables']):
#                 if visualization_element['data_settings'][i] == data_setting: 
#                     variable_data_combination = variable_name + ':' + visualization_element['data_settings'][i]
#                     compute_dict[variable_data_combination] = visualizer[variable_name]

#         j = 0
#         while j < n_samples:
#             if data_setting == 'Training':
#                 batch_x, batch_y = data_loader.train.next_batch(10)
#             results_dict = helper.sess_run_tf(sess, compute_dict, feed_dict={X: batch_x, Y: batch_y})
            
#             trace()
#             print('here')

        
#             # if variable_data_combination not in np_data_dict:

#             #     if visualization_element['data_settings'][i] == 'Training':
#             #         next_batch_func = data_loader.train.next_batch
#             #     elif visualization_element['data_settings'][i] == 'Test':
#             #         next_batch_func = data_loader.test.next_batch


#             #     visualizer[variable_name]

#             #     compute_dict = {'Input Samples': aggregate_input_images, 'Generated Samples': aggregate_transformed_sample}
#             #     batch_x, batch_y = mnist.train.next_batch(global_args.algorithm_args.batch_size*len(global_args.list_of_device_names))
#             #     curr_feed_dict = {X: batch_x, Y: batch_y}
#             #     results_dict = helper.sess_run_tf(sess, compute_dict, feed_dict=curr_feed_dict)


#             #     trace()


#             #     np_data_dict[variable_data_combination]
#             # # np_data_dict[variable_data_combination] = None
        
    
#     # results_dict = helper.sess_run_tf(sess, compute_dict, feed_dict=curr_feed_dict)


#     # trace()
