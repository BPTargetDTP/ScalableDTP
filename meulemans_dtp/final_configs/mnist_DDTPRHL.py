config = {
'beta1': 0.9,
'beta1_fb': 0.9,
'beta2': 0.99,
'beta2_fb': 0.999,
'epsilon': 3.9835298531505314e-08,
'epsilon_fb': 3.4025663897285684e-06,
'feedback_wd': 0.001621280241954684,
'lr': 2.804754262888465e-05,
'lr_fb': 0.0003024461368410027,
'sigma': 0.041623297924346575,
'target_stepsize': 0.04351591164035645,
'out_dir': 'logs/mnist/DKDTP2',
'network_type': 'DKDTP2',
'recurrent_input': False,
'hidden_fb_activation': 'tanh',
'fb_activation': 'tanh',
'initialization': 'xavier_normal',
'size_hidden_fb': 1024,
'dataset': 'mnist',
'double_precision': True,
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
'only_train_first_layer': False,
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
'log_interval': 50,
}