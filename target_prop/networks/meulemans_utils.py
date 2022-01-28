from torch import Tensor
from target_prop.networks.lenet import LeNet
from typing import Dict, Tuple
import contextlib
import io


@contextlib.contextmanager
def disable_prints():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def meulemans(
    *,
    x: Tensor,
    y: Tensor,
    network: LeNet,
    network_hparams: LeNet.HParams,
    backprop_gradients: Dict[str, Tensor],
    beta: float,
    n_pretraining_iterations: int,
    seed: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """ Minimal script to get the data of Figure 4.3 for Meuleman's DTP.

    This is to be added in the test file of `lenet_test.py`.
    """
    x = x.cuda()
    y = y.cuda()
    initial_network_weights = network.state_dict()

    from meulemans_dtp import main
    from meulemans_dtp.lib import train, builders, utils
    from meulemans_dtp.lib.conv_network import DDTPConvNetworkCIFAR
    from meulemans_dtp.final_configs.cifar10_DDTPConv import config

    # -- Get the arguments.
    # TODO: Eventually, use this typed class for the arg parsing instead of theirs. But for now,
    # we just use their code, but duck-typed using that Args class, to make it easier to use.
    # parser = main.add_command_line_args_v2()

    parser = main.add_command_line_args()
    # NOTE: They seem to want those to be strings, and then they convert stuff back to lists.
    config_with_strings = {k: str(v) if not isinstance(v, bool) else v for k, v in config.items()}
    parser.set_defaults(**config_with_strings)
    with disable_prints():
        args = parser.parse_args("")
        args = main.postprocess_args(args)

    args = typing.cast(Args, args)  # Fake cast: doesnt do anything, it's just for the type checker.

    # NOTE: Setting this, just in case they use this value somewhere I haven't seen yet.
    args.random_seed = seed
    # NOTE: Setting this to False for the LeNet equivalent network to work.
    args.freeze_BPlayers = False
    args.freeze_forward_weights = False
    # NOTE: Modifying these values so their architecture matches ours perfectly.
    args.hidden_activation = "elu"
    # NOTE: Setting beta to the same value as ours:
    args.target_stepsize = beta
    # Disable bias in their architecture if we also disable bias in ours.
    args.no_bias = not network_hparams.bias
    # Set the padding in exactly the same way as well.
    # TODO: Set this to 1 once tune-lenet is merged into master.
    DDTPConvNetworkCIFAR.pool_padding = 1
    # Set the number of iterations to match ours.
    args.nb_feedback_iterations = [n_pretraining_iterations for _ in args.nb_feedback_iterations]

    # Create their LeNet-equivalent network.
    meulemans_network = builders.build_network(args).cuda()
    assert isinstance(meulemans_network, DDTPConvNetworkCIFAR)

    # Copy over the state from our network to theirs, translating the weight names.
    missing, unexpected = meulemans_network.load_state_dict(
        translate(initial_network_weights), strict=False
    )
    assert not unexpected, f"Weights should match exactly, but got extra keys: {unexpected}."
    print(f"Arguments {missing} were be randomly initialized.")

    # Check that the two networks still give the same output for the same input, therefore that the
    # forward parameters haven't changed.
    _check_outputs_are_identical(network, meulemans_net=meulemans_network, x=x)

    meulemans_backprop_grads = get_meulemans_backprop_grads(
        meulemans_net=meulemans_network, x=x, y=y
    )
    _check_bp_grads_are_identical(
        our_network=network,
        meulemans_net=meulemans_network,
        our_backprop_grads=backprop_gradients,
        meulemans_backprop_grads=meulemans_backprop_grads,
    )

    # Q: the lrs have to be the same between the different models?
    # TODO: The network I'm using for LeNet-equivalent doesn't actually allow this to work:
    # Says "frozen blabla isn't supported with OptimizerList"
    forward_optimizer, feedback_optimizer = utils.choose_optimizer(args, meulemans_network)

    if n_pretraining_iterations > 0:
        # NOTE: Need to do the forward pass to store the activations, which are then used for feedback
        # training
        predictions = meulemans_network(x)
        train.train_feedback_parameters(
            args=args, net=meulemans_network, feedback_optimizer=feedback_optimizer
        )

    # Double-check that the forward parameters have not been updated:
    _check_forward_params_havent_moved(
        meulemans_net=meulemans_network, initial_network_weights=initial_network_weights
    )

    # Get the loss function to use (extracted from their code, was saved on train_var).
    if args.output_activation == "softmax":
        loss_function = nn.CrossEntropyLoss()
    else:
        assert args.output_activation == "sigmoid"
        loss_function = nn.MSELoss()

    # Make sure that there is nothing in the grads: delete all of them.
    meulemans_network.zero_grad(set_to_none=True)
    predictions = meulemans_network(x)
    # This propagates the targets backward, computes local forward losses, and sets the gradients
    # in the forward parameters' `grad` attribute.
    batch_accuracy, batch_loss = train.train_forward_parameters(
        args,
        net=meulemans_network,
        predictions=predictions,
        targets=y,
        loss_function=loss_function,
        forward_optimizer=forward_optimizer,
    )
    assert all(p.grad is not None for name, p in _get_forward_parameters(meulemans_network).items())

    # NOTE: the values in `p.grad` are the gradients from their DTP algorithm.
    meulemans_dtp_grads = {
        # NOTE: safe to ignore, from the check above.
        name: p.grad.detach()  # type: ignore
        for name, p in _get_forward_parameters(meulemans_network).items()
    }

    # Need to rescale these by 1 / beta as well.
    scaled_meulemans_dtp_grads = {
        key: (1 / beta) * grad for key, grad in meulemans_dtp_grads.items()
    }

    distances: Dict[str, float] = {}
    angles: Dict[str, float] = {}
    with torch.no_grad():
        for name, meulemans_backprop_grad in meulemans_backprop_grads.items():
            # TODO: Do we need to scale the DRL grads like we do ours DTP?
            meulemans_dtp_grad = scaled_meulemans_dtp_grads[name]
            distance, angle = compute_dist_angle(meulemans_dtp_grad, meulemans_backprop_grad)

            distances[name] = distance
            angles[name] = angle
        # NOTE: We can actually find the parameter for these:

    return (
        translate_back(distances, network_type=LeNet),
        translate_back(angles, network_type=LeNet),
    )

