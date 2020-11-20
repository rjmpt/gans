#contains all utility functions required for optimization

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

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

#gradient clipping
def clip_grad(grad, clip_param, mode='element'):
    if mode == 'norm':
        return tf.clip_by_norm(x, clip_param)
    elif mode == 'element':
        return tf.clip_by_value(grad, -clip_param, clip_param)

#parameter clipping
def clip_parameters(vars_list, clip_min_val=-0.01, clip_max_val=0.01):
    if len(vars_list) > 0:
        # vars_list_flat_concat = tf.concat([tf.reshape(e, [-1]) for e in vars_list], axis=0)
        # max_abs_vars_list = tf.reduce_max(tf.abs(vars_list_flat_concat))
        clip_op_list = []
        for e in cri_vars: clip_op_list.append(tf.assign(e, tf.clip_by_value(e, clip_min_val, clip_max_val)))
        #pdb.set_trace()
    return vars_list, clip_op_list

#gradient averaging
def average_gradients(grad_var_dict):
    if type(grad_var_dict) is list:
        grad_var_dict_values = grad_var_dict
    elif type(grad_var_dict) is dict:
        grad_var_dict_values = list(grad_var_dict.values())
    else: print('INVALID grad_var_dict.'); quit();

    if len(grad_var_dict_values) == 1:
        all_average_grad_and_var = grad_var_dict_values[0]
    else:
        all_average_grad_and_var = []
        all_grad_val_all_devices = zip(*grad_var_dict_values)
        for curr_grad_var_all_devices in all_grad_val_all_devices:
            # curr_grad_var_all_devices = ((grad_i_device_0, var_i_device_0),..,(grad_i_device_N, var_i_device_N))
            curr_grad_all_devices = [e[0] for e in curr_grad_var_all_devices]
            curr_var_all_devices = [e[1] for e in curr_grad_var_all_devices]
            assert (all([e == curr_var_all_devices[0] for e in curr_var_all_devices]))
            average_grad = tf_compat_v1.add_n(curr_grad_all_devices)/float(len(curr_grad_all_devices))

            average_grad_and_var = (average_grad, curr_grad_var_all_devices[0][1])
            all_average_grad_and_var.append(average_grad_and_var)
    return all_average_grad_and_var

#performs minimization using the given optimizer and loss
# Values for gate_gradients.
GATE_NONE = 0
GATE_OP = 1
GATE_GRAPH = 2
def clipped_optimizer_minimize(mode, optimizer, loss, global_step=None, var_list=None,
                               gate_gradients=GATE_OP, aggregation_method=None,
                               colocate_gradients_with_ops=False, name=None,
                               grad_loss=None, clip_param=None):

    if mode == 'Adam' or mode == 'RMSProp':
        grads_and_vars = optimizer.compute_gradients(
            loss, var_list=var_list, gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss)

        if clip_param is not None and clip_param > 0:
            clipped_grads_and_vars = [(clip_grad(grad, clip_param), var) if grad is not None else (grad, var) for grad, var in grads_and_vars]
        else: clipped_grads_and_vars = grads_and_vars

        clipped_grads_and_vars = [(g, v) for g, v in clipped_grads_and_vars if g is not None]
        if len(clipped_grads_and_vars) == 0:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops"
                " that do not support gradients, between variables %s and loss %s." %
                ([str(v) for _, v in clipped_grads_and_vars], loss))
        
        return clipped_grads_and_vars
        
    elif mode == 'SGLD':
        optimizer.iterations = global_step

        grads = tf.gradients(loss, var_list)
        grads_and_vars = zip(grads, var_list)
        if clip_param is not None and clip_param > 0:
            clipped_grads_and_vars = [(tf.clip_by_norm(grad, clip_param), var) if grad is not None else (grad, var) for grad, var in grads_and_vars]
        elif clip_param < 0: pdb.set_trace()
        else: clipped_grads_and_vars = grads_and_vars

        vars_with_grad = [v for g, v in clipped_grads_and_vars if g is not None]

        if not vars_with_grad:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops"
                " that do not support gradients, between variables %s and loss %s." %
                ([str(v) for _, v in clipped_grads_and_vars], loss))

        return clipped_grads_and_vars
        
#optimizer wrapper
def get_optimizer(epoch_step, optimizer_class, learning_rate, learning_rate_decay_rate, beta1, beta2, epsilon, max_epochs):
    if optimizer_class == 'Adam':
        return tf_compat_v1.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=1e-08)    
    
    elif optimizer_class == 'AdamWithDecay':
        curr_learning_rate = tf_compat_v1.train.polynomial_decay(learning_rate=learning_rate, global_step=epoch_step, 
                             decay_steps=max_epochs-1, end_learning_rate=learning_rate*learning_rate_decay_rate, power=2.)
        return tf_compat_v1.train.AdamOptimizer(learning_rate=curr_learning_rate, beta1=beta1, beta2=beta2, epsilon=1e-08)
    
    elif optimizer_class == 'SGLD':
        curr_learning_rate = tf_compat_v1.train.polynomial_decay(learning_rate=learning_rate, global_step=epoch_step, 
                             decay_steps=max_epochs-1, end_learning_rate=learning_rate*learning_rate_decay_rate, power=2.)
        return tfp.optimizer.StochasticGradientLangevinDynamics(learning_rate=curr_learning_rate, preconditioner_decay_rate=beta2)

#schedules optimization based on a frequency dict if different networks in a model need to be optimized at different frequencies (such as in wGAN)
class OptimizationScheduler():
    def __init__(self, opt_freq_dict, registry_list, mode='First', verbose=False):
        self.opt_freq_dict = opt_freq_dict
        self.mode = mode
        assert (mode == 'First' or mode == 'Last' or mode == 'Spaced')

        self.schedule_opts()
        self.register_dicts(registry_list)
        if verbose: self.print_schedule()

    def print_schedule(self):
        print('\n\n\nOptimization Scheduler: Period = '+str(self.period))
        for i, e in enumerate(self.included_opt_dict_list): 
            print('Phase (' + str(i) + '): ' + str(e)+ ' | Total Groups: '+ str(np.sum(np.asarray(list(e.values())) == True)))
        print('\n\n\n')

    def get_included_ops(self, freq, max_freq):
        if self.mode == 'First':
            return list(range(freq))
        elif self.mode == 'Last':
            return list(range(max_freq-freq, max_freq))
        else:
            include_list = [0]
            for i in range(1, freq):
                include_list.append(int(i * (max_freq-1) / (freq-1)))
            return include_list

    def register_dicts(self, name_dict_pair_list):
        self.registered_opt_dict_list = []
        for phase in range(self.period):
            phase_bool_dict = self.get_opt_bool_dict(phase)
            curr_overall_dict = {}
            for (name, dict_element) in name_dict_pair_list:
                curr_overall_dict = {**curr_overall_dict, 
                    **{name + '_' + key: dict_element[key] for key in phase_bool_dict if phase_bool_dict[key]}}
            self.registered_opt_dict_list.append(curr_overall_dict)

    def schedule_opts(self):
        k = list(self.opt_freq_dict.keys())
        v = list(self.opt_freq_dict.values())

        max_value = max(v)
        self.period = max_value
        max_key = k[v.index(max_value)]

        self.included_opt_dict_list = [dict.fromkeys(k, False) for i in range(max_value)]

        for key in k:
            true_list = self.get_included_ops(self.opt_freq_dict[key], max_value)
            for i in true_list:
                self.included_opt_dict_list[i][key] = True
    
    def get_opt_bool_dict(self, i):
        self.phase = i % self.period
        return self.included_opt_dict_list[self.phase]

    def get_scheduled_registered_dict(self, i):
        self.phase = i % self.period
        return self.registered_opt_dict_list[self.phase]
