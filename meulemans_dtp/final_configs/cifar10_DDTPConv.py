config = {
'beta1': 0.9,
'beta1_fb': 0.9,
'beta2': 0.999,
'beta2_fb': 0.999,
'epsilon': '[2.7867895625009e-08, 1.9868935703787622e-08, 4.515242618159344e-06, 4.046144976139705e-05]',
'epsilon_fb': 7.529093372180766e-07,
'feedback_wd': 6.169295107849636e-05,
'lr': [0.00025935571806476586, 0.000885500279951265, 0.0001423047695105589, 3.3871035558126015e-06],
'lr_fb': 0.0045157498494467095,
'nb_feedback_iterations': [1, 1, 1, 1],
'sigma': 0.00921040366516759,
'target_stepsize': 0.015962099947441903,
'dataset': 'cifar10',
#'out_dir': 'logs/DDTPConv_cifarCIFAR',
'network_type': 'DDTPConvCIFAR',
'initialization': 'xavier_normal',
'fb_activation': 'linear',

# ### Training options ###
'optimizer': 'Adam',
'optimizer_fb': 'Adam',
'momentum': 0.,
'parallel': True,
'normalize_lr': True,
'batch_size': 128,
'epochs_fb': 10,
'not_randomized': True,
'not_randomized_fb': True,
'extra_fb_minibatches': 0,
'extra_fb_epochs': 1,
'epochs': 90,
'double_precision': True,

### Network options ###
# 'num_hidden': 3,
# 'size_hidden': 1024,
# 'size_input': 3072,
# 'size_output': 10,
'hidden_activation': 'tanh',
'output_activation': 'softmax',
'no_bias': False,

### Miscellaneous options ###
'no_cuda': False,
#'random_seed': 42,
'cuda_deterministic': False,
'freeze_BPlayers': False,
'multiple_hpsearch': False,

### Logging options ###
'save_logs': False,
'save_BP_angle': False,
'save_GN_angle': False,
'save_GN_activations_angle': False,
'save_BP_activations_angle': False,
'gn_damping': 0.,
'log_interval': 100,
}
