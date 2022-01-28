config = {
'lr': 5.800132985181198e-05,
'target_stepsize': 0.010956946694971827,
'feedback_wd': 5.325034729359803e-05,
'beta1': 0.9,
'beta2': 0.99,
'epsilon': 7.505830224780078e-06,
'lr_fb': 0.0002003429090524803,
'sigma': 0.0825200614614339,
'beta1_fb': 0.99,
'beta2_fb': 0.99,
'epsilon_fb': 3.459367678282587e-05,
'out_dir': 'logs/mnist/DTP_improved',
'network_type': 'DTP',
'initialization': 'xavier_normal',
'dataset': 'mnist',
'double_precision': True,
'optimizer': 'Adam',
'optimizer_fb': 'Adam',
'momentum': 0.0,
'parallel': True,
'normalize_lr': True,
'batch_size': 128,
'forward_wd': 0.0,
'epochs_fb': 10,
'not_randomized': True,
'not_randomized_fb': True,
'extra_fb_minibatches': 0,
'extra_fb_epochs': 0,
'epochs': 100,
'only_train_first_layer': True,
'train_only_feedback_parameters': False,
'num_hidden': 5,
'size_hidden': 256,
'size_input': 784,
'size_output': 10,
'hidden_activation': 'tanh',
'output_activation': 'softmax',
'no_bias': False,
'no_cuda': False,
'random_seed': 42,
'cuda_deterministic': False,
'freeze_BPlayers': False,
'multiple_hpsearch': False,
'save_logs': False,
'save_BP_angle': False,
'save_GN_angle': False,
'save_GN_activations_angle': False,
'save_BP_activations_angle': False,
'gn_damping': 0.0,
'hpsearch': False,
'log_interval': 30,
}