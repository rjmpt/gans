#this is the primary file in which experiments are actually executed

from __future__ import print_function
from IPython.core.debugger import Pdb
pdb = Pdb()
trace = pdb.set_trace

import datetime
curr_date_string = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
import os
import sys
import inspect
import traceback
import gc
import shutil
import argparse
import time
import random
from pathlib import Path

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
import optimization
import visualization

parser = argparse.ArgumentParser(description='')
parser.add_argument('--experiment_mode', type=str, default='Training', help="Options: 'Debug', 'Training', 'Deployment'.")
parser.add_argument('--main_experiments_dir', type=str, default=str(Path.home())+'/Experiments/', help="Main experiments directory.")
parser.add_argument('--experiment_group', type=str, default='671_final_proj', help="Experiment group.")


parser.add_argument('--restore_dir', type=str, default='/home/adi/hdd3/Experiments/671_final_proj/wGAN_gp-DCGAN/CIFAR10/1a5b8efc0c0b405994d4bde69d51bce8/', help="Restore model directory.")
parser.add_argument('--restore', type=bool, default=False, help="Restore checkpoint?.")
parser.add_argument('--seed', type=int, default=31232, help="Random seed.")
parser.add_argument('--max_batches_per_epoch', type=int, default=10000, help="Maximum number of batches to include in 1 epoch.")
parser.add_argument('--max_epochs', type=int, default=200, help="Maximum number of epochs to run the optimization.")
parser.add_argument('--training_log_freq', type=int, default=10, help="Training log frequency (number of times per epoch).")
parser.add_argument('--max_hyperparameter_epochs', type=int, default=0, help="Maximum number of epochs to run the hyperparameter optimization.")
parser.add_argument('--algorithm_name', type=str, default='fGAN-Pearson-DCGAN', help="The name of the algorithm to experiment with.")
parser.add_argument('--dataset_name', type=str, default='CIFAR10', help="The name of the dataset to experiment with.")
parser.add_argument('--gpu_ID_list', type=list, default=['0'], help="Options: [], ['0'], ['1'], ['0', '1'], ['1', '0']")

#parse in all the experiment arguments
global_args = parser.parse_args()
global_args.temporal_dict = helper.AttrDict()
global_args.temporal_dict.epoch = 0
global_args.temporal_dict.hyperparameter_epoch = 0
global_args.temporal_dict.data_epoch = 0

#set up random seeds
random.seed(global_args.seed)
np.random.seed(global_args.seed)
global_args.parameter_device_name, global_args.list_of_device_names = helper.get_device_names(global_args.gpu_ID_list)
if global_args.experiment_mode == 'Debug': tf.debugging.set_log_device_placement(True)
elif global_args.experiment_mode == 'Training' or global_args.experiment_mode == 'Deployment':
    global_args.exp_dir = helper.make_exp_dir(global_args)

#get the model and hyperparameter details for the experiment
Model, global_args.algorithm_args, global_args.optimization_args = helper.get_model_and_algorithm_optimization_args(global_args.algorithm_name)
data_loader = helper.get_data_loader(global_args)
_, _, example_batch = next(data_loader)

#set up tf graph
tf_graph = tf.Graph()
visualizer = helper.AttrDict()
with tf_graph.as_default():
    tf_compat_v1.set_random_seed(global_args.seed)
    model = Model(global_args=global_args, config=global_args.algorithm_args) #set up model

    with tf.device(global_args.parameter_device_name):
        tf_input_dict, feed_dict_func = helper.tf_replicate_batch_dict(example_batch, global_args.list_of_device_names, global_args.algorithm_args.batch_size)
        
        global_step = tf_compat_v1.Variable(0, dtype=tf.int64, name='global_step', trainable=False) # increments once per update of parameters.
        epoch_step = tf_compat_v1.Variable(0, dtype=tf.int64, name='epoch_step', trainable=False) # increments once per epochs of data seen.
        increment_global_step_op = tf_compat_v1.assign(global_step, global_step+1)
        increment_epoch_step_op = tf_compat_v1.assign(epoch_step, epoch_step+1)

        #collect all the optimizers in a dictionary
        optimizers_dict = {}
        for group in global_args.optimization_args:
            group_args = global_args.optimization_args[group]
            optimizers_dict[group] = optimization.get_optimizer(epoch_step=epoch_step, optimizer_class=group_args.optimizer_class, 
                learning_rate=group_args.learning_rate, learning_rate_decay_rate=group_args.learning_rate_decay_rate, 
                beta1=group_args.beta1, beta2=group_args.beta2, epsilon=group_args.epsilon, max_epochs=global_args.max_epochs)

    #perform training across multiple GPUs or multiple CPU/GPU as specified
    grads_and_vars_all_groups_all_devs, optimized_vars = {}, {}
    for i, curr_device_name in enumerate(global_args.list_of_device_names):
        with tf.device(helper.assign_to_device(device_name=curr_device_name, parameter_device_name=global_args.parameter_device_name)):
            model.inference(tf_input_dict['per_device'][curr_device_name], device_name=curr_device_name)
            helper.set_grads_and_vars_for_device(grads_and_vars_all_groups_all_devs, optimized_vars, 
                curr_device_name, model.computations, optimizers_dict, global_step, global_args.optimization_args)
            
    #aggregate gradients and average them
    with tf.device(global_args.parameter_device_name): 
        train_step_tf_dict, objective_tf_dict = {}, {}
        for group in grads_and_vars_all_groups_all_devs:
            train_step_tf_dict[group] = optimizers_dict[group].apply_gradients(optimization.average_gradients(grads_and_vars_all_groups_all_devs[group]))            
            objective_tf_dict[group] = helper.tf_average_n([model.computations[curr_device_name].objective_dict[group] for curr_device_name in model.computations])
        
    #set things up to ensure automatic termination if we are running out of memory
    memory_node = None
    if len(global_args.gpu_ID_list) > 0: memory_node = tf.contrib.memory_stats.MaxBytesInUse()
    init = tf_compat_v1.global_variables_initializer()
    saver = tf_compat_v1.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=10000, pad_step_number=True)
    
    config = tf_compat_v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf_compat_v1.Session(config=config)
    merged_summaries = tf_compat_v1.summary.merge_all()
    summary_writer = tf_compat_v1.summary.FileWriter(global_args.exp_dir+'summaries/', sess.graph)
    sess.run(init)

    if global_args.restore:
        print('=> Loading checkpoint: ' + global_args.restore_dir)
        epoch_from_checkpoint, global_args_from_checkpoint = helper.load_checkpoint(saver, sess, global_args.restore_dir, load_meta_graph=True) 
        global_args.temporal_dict.epoch = epoch_from_checkpoint+1
        global_args.temporal_dict.data_epoch = epoch_from_checkpoint+1
        global_args.exp_dir = global_args.restore_dir
        
def hyperparameter_training():
    return

#automatically generate visualizations and save them all in the experiment directory
def visualize_results():
    visualization.visualize_model(sess, model, data_loader, feed_dict_func, tf_graph, global_args.parameter_device_name, global_args.temporal_dict, global_args.exp_dir)

#train the model!
def model_training():
    data_loader.setup('Training', randomized=True) #data loader

    #grab all the optimizers
    optimization_groups = list(train_step_tf_dict.keys())
    log_interval = int(np.floor(float(data_loader.curr_max_iter)/(float(global_args.training_log_freq))))
    acc_training_objectives = helper.AccumAttrDict(['Objective' + '_' + e for e in optimization_groups])
    optimization_scheduler = optimization.OptimizationScheduler(opt_freq_dict=global_args.algorithm_args.group_freq_dict,
        registry_list=[('Objective', objective_tf_dict), ('Step', train_step_tf_dict)], mode='Last', verbose=True)

    #process all the batches
    for i, curr_batch_size, batch_np in data_loader: 
        t_start = time.time()
        compute_dict = optimization_scheduler.get_scheduled_registered_dict(i) #get computation dict corresponding to this batch
        results_dict = helper.sess_run_tf(sess, compute_dict, feed_dict=feed_dict_func(batch_np))
        acc_training_objectives.add_values(results_dict, weight_size_spec=curr_batch_size)
        sess.run(increment_global_step_op) 
        t_end = time.time()

        #log as necessary
        if i % log_interval == 0:
            acc_results_dict = acc_training_objectives.normalize()
            print_str = ''
            if 'Objective_Critic' in results_dict: print_str += '  (Objective) Critic: ' + "{:.3f}".format(results_dict['Objective_Critic'])
            if 'Objective_Generator' in results_dict: print_str += '  (Objective) Generator: ' + "{:.3f}".format(results_dict['Objective_Generator'])
            if 'Objective_Classifier' in results_dict: print_str += '  (Objective) Classifier: ' + "{:.3f}".format(results_dict['Objective_Classifier'])

            if 'Objective_Critic' in acc_results_dict: print_str += '  ACC (Objective) Critic: ' + "{:.3f}".format(acc_results_dict['Objective_Critic'])
            if 'Objective_Generator' in acc_results_dict: print_str += '  ACC (Objective) Generator: ' + "{:.3f}".format(acc_results_dict['Objective_Generator'])
            if 'Objective_Classifier' in acc_results_dict: print_str += '  ACC (Objective) Classifier: ' + "{:.3f}".format(acc_results_dict['Objective_Classifier'])

            print('Epoch/Update: ' + str((global_args.temporal_dict.epoch, i)) + '  Opt. time for current iteration: '+"{:.3f}".format(t_end-t_start) + print_str)

print('\n\n\n\n\n\nStarting experiment:\n\n\n')

#model inference / deployment
if global_args.experiment_mode == 'Deployment':
    assert (global_args.restore)
    print('\n\n\nStarting deployment of a previously trained model.\n\n\n')
    # DO DEPLOYMENT OF A MODEL
    gc.collect(); gc.collect()
    print('=== Memory After visualize(): ' + helper.report_memory(sess, memory_node) + '\n\n')
    print('Finished deployment.')

#training
elif global_args.experiment_mode == 'Training':
    #hyperparam optimization
    if global_args.max_hyperparameter_epochs > 0:
        print('\nStarting hyperparameter training.\n\n')  
        while global_args.temporal_dict.hyperparameter_epoch <= global_args.max_hyperparameter_epochs:
            print('=== Memory usage at start of hyperparameter training: ' + helper.report_memory(sess, memory_node) + '\n\n')
            hyperparameter_training()
            gc.collect(); gc.collect()
            print('=== Memory usage after hyperparameter training: ' + helper.report_memory(sess, memory_node) + '\n\n')
            global_args.temporal_dict.hyperparameter_epoch += 1 

    print('\nStarting model parameter training.\n\n')   

    #model parameter training
    while global_args.temporal_dict.epoch <= global_args.max_epochs + 1:
        print('=== Memory usage at start of visualizing model: ' + helper.report_memory(sess, memory_node) + '\n\n')
        visualize_results()
        gc.collect(); gc.collect()
        print('=== Memory usage after visualizing model: ' + helper.report_memory(sess, memory_node) + '\n\n')

        print('=== Memory usage at start of model training: ' + helper.report_memory(sess, memory_node) + '\n\n')
        model_training()
        gc.collect(); gc.collect()
        print('=== Memory usage after model training: ' + helper.report_memory(sess, memory_node) + '\n\n')

        print('Experiment Directory: '+ global_args.exp_dir)

        helper.save_checkpoint(saver, sess, global_args, global_step, global_args.exp_dir, save_meta_graph=True) #checkpoint after every epoch

        sess.run(increment_epoch_step_op) 
        global_args.temporal_dict.epoch += 1 
        global_args.temporal_dict.data_epoch += 1 
  







