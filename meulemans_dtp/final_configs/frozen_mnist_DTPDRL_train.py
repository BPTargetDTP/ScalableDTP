config = {
'lr': 2.9397069314888417e-05,
'target_stepsize': 0.04214703353466974,
'beta1': 0.9,
'beta2': 0.99,
'epsilon': 4.2708694335174877e-07,
'lr_fb': 0.0009259703434043203,
'sigma': 0.057544149745474386,
'feedback_wd': 4.726104223717957e-07,
'beta1_fb': 0.9,
'beta2_fb': 0.999,
'epsilon_fb': 5.688987601546281e-06,
'out_dir': 'logs/mnist/DTPDR',
'network_type': 'DTPDR',
'initialization': 'xavier_normal',
'fb_activation': 'tanh',
'dataset': 'mnist',
'optimizer': 'Adam',
'optimizer_fb': 'Adam',
'momentum': 0.0,
'parallel': True,
'normalize_lr': True,
'batch_size': 128,
'forward_wd': 0.0,
'epochs_fb': 6,
'not_randomized': True,
'not_randomized_fb': True,
'extra_fb_minibatches': 0,
'extra_fb_epochs': 1,
'epochs': 100,
'only_train_first_layer': True,
'train_only_feedback_parameters': False,
'no_val_set': True,
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
'double_precision': True,
'save_logs': False,
'save_BP_angle': False,
'save_GN_angle': False,
'save_GN_activations_angle': False,
'save_BP_activations_angle': False,
'gn_damping': 0.0,
'hpsearch': False,
'log_interval': 80,
}
