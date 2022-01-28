"""
TODOs:
- [X] Make sure we both use the same number of iterations (very large)?
    - Or use an improvement criterion?
- [ ] Try changing our architecture to match theirs as well.
- [x] Set eta (our beta) for meulemans to the same value as us.
- [x]: Normalize the updates by beta as well
- [X]: Remove the bias from fig 4.3
- [ ]: Change beta, see if that makes it any better: 1e-4
- [ ]: Maybe test different normalizations?
- [X]: Make angle / distance figure for SimpleVGG
"""
import functools
from argparse import Namespace
from typing import (
    Callable,
    Dict,
    Mapping,
    Optional,
    OrderedDict,
    Tuple,
    Type,
    Union,
)

import torch
from main_pl import Model
from target_prop.models import BaselineModel, DTP, Model, baseline
from target_prop.models.dtp import DTP
from target_prop.networks import LeNet, Network, SimpleVGG
from target_prop.networks.simple_vgg import SimpleVGG
from target_prop.scheduler_config import StepLRConfig
from torch import Tensor, nn

try:
    import meulemans_dtp.main
except ImportError as e:
    raise RuntimeError(
        "You need the submodule of the meulemans DTP repo initialized to run this script. \n"
        "Run `git submodule init` and `git submodule update` and try again."
    )
from meulemans_dtp.main import Args
from meulemans_dtp.lib.conv_network import DDTPConvNetworkCIFAR
from meulemans_dtp.lib.direct_feedback_networks import DTPDRLNetwork
from meulemans_dtp.lib import builders
from meulemans_dtp.lib.direct_feedback_networks import DTPDRLNetwork
from meulemans_dtp.lib.networks import DTPNetwork


# TODO: Update this list with the best known hyper-parameters (model and algo).
_best_hparams: Mapping[
    Tuple[str, Type[Model], Type[Network]], Tuple[Model.HParams, Network.HParams]
] = {  # type: ignore
    ("cifar10", DTP, SimpleVGG): (DTP.HParams(), SimpleVGG.HParams()),
    # TODO: Get these from anonymous.
    ("cifar10", DTP, LeNet): (
        # NOTE: https://wandb.ai/anonymous/scalingDTP/runs/25evwi2d?workspace=user-anonymous
        # main_pl.py run dtp lenet --seed 124
        # --beta 0.46550286113514694 --noise 0.41640228838517584 0.3826261146623929 0.13953820693586017
        # --scheduler True --b_optim.lr 0.0007188427494432325 0.00012510321884615596 0.03541466958291287
        # --batch_size 144 --f_optim.lr 0.03618358843718276 --max_epochs 90 --dataset cifar10
        # --f_optim.type sgd --b_optim.type sgd --channels 32 64 --activation elu
        # --early_stopping_patience 0 --feedback_training_iterations 41 51 24
        # --feedback_samples_per_iteration 1 --data_dir /Tmp/slurm.1453692.0
        DTP.HParams.from_dict(
            dict(
                beta=0.46550286113514694,
                noise=[0.41640228838517584, 0.3826261146623929, 0.13953820693586017],
                use_scheduler=True,
                b_optim=dict(
                    type="sgd",
                    lr=[0.0007188427494432325, 0.00012510321884615596, 0.03541466958291287,],
                ),
                batch_size=144,
                f_optim=dict(type="sgd", lr=0.03618358843718276),
                max_epochs=90,
                early_stopping_patience=0,
                feedback_training_iterations=[41, 51, 24],
                feedback_samples_per_iteration=1,
            )
        ),
        LeNet.HParams(channels=[32, 64], activation=nn.ELU),
        # DTP.HParams.from_run(
        #     "anonymous/scalingDTP/25evwi2d",
        #     renamed_keys={"scheduler": "use_scheduler"},
        #     cache_dir=CACHE_DIR,
        # ),
        # LeNet.HParams.from_run("anonymous/scalingDTP/25evwi2d", cache_dir=CACHE_DIR,),
    ),
    ("cifar10", BaselineModel, SimpleVGG): (
        # TODO: Do something smarter, e.g. this:
        # best_run_from("anonymous/scalingDTP", where="filter expression.", metric="val/accuracy")
        BaselineModel.HParams(
            batch_size=28,
            early_stopping_patience=0,
            f_optim=baseline.ForwardOptimizerConfig(
                type="sgd", momentum=0.9, lr=0.001661, weight_decay=0.0001
            ),
            lr_scheduler=StepLRConfig(frequency=1, gamma=0.4898, interval="epoch", step_size=49),
            max_epochs=90,
            use_scheduler=False,
        ),
        SimpleVGG.HParams(activation=nn.ELU, channels=[128, 128, 256, 256, 512]),
        # BaselineModel.HParams.from_run(
        #     "anonymous/scalingDTP/3shnwuf8", removed_keys=["hp/channels", "hp/activation"]
        # ),
        # SimpleVGG.HParams.from_run(
        #     "anonymous/scalingDTP/3shnwuf8",
        #     # renamed_keys={"hp/channels": "net_hp/channels"},
        # ),
    ),
}


import inspect
from typing import TypeVar

import meulemans_dtp.lib.conv_network
import meulemans_dtp.lib.direct_feedback_networks
import meulemans_dtp.lib.networks
from meulemans_dtp.lib import conv_network, direct_feedback_networks, networks
from typing_extensions import ParamSpec

T = TypeVar("T")

P = ParamSpec("P")

# NOTE: Making `fixed_args` and `**fixed_kwargs` below with P.args and P.kwargs makes the type
# checkers give an error when we don't pass ALL the parameters, which isn't exactly what I want
# here.
# What I want is some way of indicating that, when passed, these need to be valid args or kwargs to
# the callable `cls`.
def with_fixed_constructor_arguments(
    cls: Callable[P, T], *fixed_args, **fixed_kwargs,
) -> Callable[P, T]:
    """ Returns a callable that fixes some of the arguments to the type or callable `cls`.
    """
    if not fixed_args and not fixed_kwargs:
        # Not fixing any arguments, return the callable as-is.
        return cls
    # NOTE: There's apparently no need to pass cls.__init__ for classes. So we can do the same for
    # either classes or functions.
    init_signature = inspect.signature(cls)
    try:
        bound_fixed_args = init_signature.bind_partial(*fixed_args, **fixed_kwargs)
    except TypeError as err:
        raise TypeError(f"Unable to bind fixed values for {cls}: {err}") from err

    @functools.wraps(cls)
    def _wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        bound_args = init_signature.bind_partial(*args, **kwargs)
        for argument_name, fixed_value in bound_fixed_args.arguments:
            print(
                f"Ignoring value {bound_args.arguments[argument_name]} for argument "
                f"{argument_name}, using fixed value of {fixed_value} instead."
            )
        bound_args.arguments.update(bound_fixed_args.arguments)
        bound_args.apply_defaults()
        return cls(*bound_args.args, **bound_args.kwargs)

    return _wrapped


import meulemans_dtp.final_configs


def get_meulemans_args_for(dataset: str, our_network_class: Type[Network]) -> Args:
    """ Get the `args` object to use for the given dataset and desired network type.

    This `arg` object is used throughout the Meulemans DTP codebase, and contains all the
    hyper-parameters, configuration options, etc.
    """
    assert our_network_class is LeNet, "todo: only have a LeNet-like network in their codebase."
    assert dataset == "cifar10", "todo: debugging, remove this later"
    config: Optional[Dict] = None
    if dataset == "cifar10":
        # TODO: Double-check this with @anonymous
        if our_network_class is LeNet:
            from meulemans_dtp.final_configs.cifar10_DDTPConv import config

            # from meulemans_dtp.final_configs. import config
        elif our_network_class is SimpleVGG:
            from meulemans_dtp.final_configs.cifar10_DDTPlinear import config

    if config is None:
        raise NotImplementedError(
            f"Don't yet know which config file from the Meulemans DTP repo to use for dataset "
            f"{dataset} and desired network type {our_network_class}"
        )
    overwrite_defaults = config
    # NOTE: They seem to want those to be strings?
    overwrite_defaults = {k: str(v) if not isinstance(v, bool) else v for k, v in config.items()}
    parser = meulemans_dtp.main.add_command_line_args()
    if overwrite_defaults is not None:
        parser.set_defaults(**overwrite_defaults)
    args = parser.parse_args("")
    args = meulemans_dtp.main.postprocess_args(args)
    return args


from meulemans_dtp.lib import builders
from target_prop.models import DTP
from target_prop.networks import LeNet


def create_meulemans_network(
    dataset: str, network_type: Type[Network], initial_network_weights: Dict[str, Tensor],
) -> Tuple[Namespace, DDTPConvNetworkCIFAR]:
    # def get_meulemans_grad_distances_and_angles(seed: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    # note; need to create this dummy args thingy.
    args = get_meulemans_args_for(dataset=dataset, our_network_class=network_type)
    # NOTE: Setting this, just in case they use that value somewhere I haven't seen yet.
    args.random_seed = 123
    # NOTE: Setting this to False for the LeNet equivalent network to work.
    args.freeze_BPlayers = False
    meulemans_net = builders.build_network(args)
    assert isinstance(meulemans_net, DDTPConvNetworkCIFAR), "only works on this model atm."
    meulemans_net = meulemans_net.cuda()

    assert network_type is LeNet, "Only know this mapping for the LeNet atm."

    # Translate the keys of the initial parameter dict, so we can load state dict with it:
    meulemans_net_initial_parameters = translate(initial_network_weights)

    # Reload the state-dict, so that the parameters are exactly the same for our DTP and theirs!
    # NOTE: There are some missing keys: the params of the feedback networks.
    # IDEA: What about also loading the state dict of the feedback weights? Ideally we'd like to do
    # that, if possible, right?
    missing_keys, unexpected_keys = meulemans_net.load_state_dict(
        meulemans_net_initial_parameters, strict=False
    )
    assert not unexpected_keys, "We should have loaded all the forward params of their net."
    print(f"Parameters {missing_keys} were randomly initialized.")
    # TODO: Add checks for all other conditions / params, to make sure they are equivalent.
    # assert args.fb_activation

    return args, meulemans_net


def translate(dtp_values: Dict[str, T]) -> OrderedDict[str, T]:
    """ Translate our network param names to theirs. """
    return OrderedDict(
        [(our_network_param_names_to_theirs[LeNet][k], v) for k, v in dtp_values.items()]
    )


def translate_back(
    meulemans_values: Dict[str, T], network_type: Type[Network] = LeNet
) -> Dict[str, T]:
    """ Translate thir network param names back to ours. """
    return {
        their_network_param_names_to_ours[network_type][k]: v for k, v in meulemans_values.items()
    }


def _get_forward_parameters(meulemans_net) -> Dict[str, Tensor]:
    # NOTE: Gets the forward weights dict programmatically.
    # The networks only return them as a list normally.
    meulemans_net_forward_params_list = meulemans_net.get_forward_parameter_list()
    # Create a dictionary of the forward parameters by finding the matching entries:
    return {
        name: param
        for name, param in meulemans_net.named_parameters()
        for forward_param in meulemans_net_forward_params_list
        if param is forward_param
    }


def _check_forward_params_havent_moved(
    meulemans_net: DDTPConvNetworkCIFAR, initial_network_weights: Dict[str, Tensor]
):
    # Translate the keys of the initial parameter dict, so we can load state dict with it:
    meulemans_net_initial_parameters = translate(initial_network_weights)
    meulemans_net_forward_params = _get_forward_parameters(meulemans_net)
    for name, parameter in meulemans_net_forward_params.items():
        assert name in meulemans_net_initial_parameters
        initial_value = meulemans_net_initial_parameters[name]
        # Make sure that the initial value wasn't somehow moved into the model, and then modified
        # by the model.
        assert parameter is not initial_value
        # Check that both are equal:
        if not torch.allclose(parameter, initial_value):
            raise RuntimeError(
                f"The forward parameter {name} was affected by the feedback training?!",
                (parameter - initial_value).mean(),
            )


def _check_outputs_are_identical(
    our_network: Network, meulemans_net: DDTPConvNetworkCIFAR, x: Tensor
):
    # Check that their network gives the same output for the same input as ours.
    # NOTE: No need to run this atm, because they use tanh, and we don't. So this won't match
    # anyway.
    rng_state = torch.random.get_rng_state()
    our_output: Tensor = our_network(x)
    torch.random.set_rng_state(rng_state)
    their_output: Tensor = meulemans_net(x)
    meulemans_forward_params = translate_back(
        _get_forward_parameters(meulemans_net), network_type=type(our_network)
    )
    for param_name, param in our_network.named_parameters():
        their_param = meulemans_forward_params[param_name]
        if not torch.allclose(param, their_param):
            raise RuntimeError(
                f"Weights for param {param_name} aren't the same between our model and Meulemans', "
                f" so the output won't be the same!"
            )

    our_x = x
    their_x = x.clone()  # just to be 200% safe.

    assert len(our_network) == len(meulemans_net.layers)

    for layer_index, (our_layer, their_layer) in enumerate(zip(our_network, meulemans_net.layers)):
        our_x = our_layer(our_x)
        # In their case they also need to flatten here.
        if layer_index == meulemans_net.nb_conv:
            their_x = their_x.flatten(1)
        their_x = their_layer(their_x)

        # their_x = their_layer(their_x)
        if our_x.shape != their_x.shape:
            raise RuntimeError(
                f"Output shapes for layer {layer_index} don't match! {our_x.shape=}, "
                f"{their_x.shape=}!"
            )

        if not torch.allclose(our_x, their_x):
            # breakpoint()
            raise RuntimeError(f"Output of layers at index {layer_index} don't match!")

    if not torch.allclose(our_output, their_output):
        raise RuntimeError(
            f"The Meulamans network doesn't produce the same output as ours!\n"
            f"\t{our_output=}\n"
            f"\t{their_output=}\n"
            f"\t{(our_output - their_output).abs().sum()=}\n"
        )


def _check_bp_grads_are_identical(
    our_network: Network,
    meulemans_net: DDTPConvNetworkCIFAR,
    our_backprop_grads: Dict[str, Tensor],
    meulemans_backprop_grads: Dict[str, Tensor],
):
    # Compares our BP gradients and theirs: they should be identical.
    # If not, then there's probably a difference between our network and theirs (e.g. activation).
    # NOTE: Since the activation is different, we don't actually run this check.
    for param_name, dtp_bp_grad in our_backprop_grads.items():
        meulemans_param_name = our_network_param_names_to_theirs[type(our_network)][param_name]
        meulemans_bp_grad = meulemans_backprop_grads[meulemans_param_name]
        if meulemans_bp_grad is None:
            if dtp_bp_grad is not None:
                raise RuntimeError(
                    f"Meulemans DTP doesn't have a backprop grad for param {param_name}, but our "
                    f"DTP model has one!"
                )
            continue
        assert meulemans_bp_grad.shape == dtp_bp_grad.shape
        if not torch.allclose(dtp_bp_grad, meulemans_bp_grad):
            raise RuntimeError(
                f"Backprop gradients for parameter {param_name} aren't the same as ours!\n"
                f"\tTheir backprop gradient:\n"
                f"\t{dtp_bp_grad}\n"
                f"\tTheir backprop gradient:\n"
                f"\t{meulemans_bp_grad}\n"
                f"\tTotal absolute differentce:\n"
                f"\t{(dtp_bp_grad - meulemans_bp_grad).abs().sum()=}\n"
            )


# Maps the names of the networks in the Meulemans DTP codebase to the classes that they map to.
# NOTE: This was extracted from meulemans_dtp/lib/builders.py
meulemans_dtp_network_names_to_classes: Dict[
    str, Union[Type[DTPNetwork], Type[conv_network.DDTPConvNetwork], Callable[..., nn.Module]],
] = {
    "DTPDRL": DTPDRLNetwork,
    # "LeeDTP": meulemans_dtp.lib.networks.LeeDTPNetwork,
    "DTP": networks.DTPNetwork,
    "DTPDR": networks.DTPDRLNetwork,
    "DKDTP2": direct_feedback_networks.DDTPRHLNetwork,
    "DMLPDTP2": direct_feedback_networks.DDTPMLPNetwork,
    "DDTPControl": direct_feedback_networks.DDTPControlNetwork,
    "GN2": networks.GNTNetwork,
    "BP": networks.BPNetwork,
    "DDTPConv": conv_network.DDTPConvNetwork,
    # NOTE: would reset the value back to `DDTPConv`, which seems to imply that other parts of the
    # codebase are not able to handle this value, but that we still want a specialized network for
    # cifar10.
    # This is the textbook use-case for inheritance IMO.
    "DDTPConvCIFAR": conv_network.DDTPConvNetworkCIFAR,
    # args.network_type = 'DDTPConv'
    "DDTPConvControlCIFAR": conv_network.DDTPConvControlNetworkCIFAR,
    # args.network_type = 'DDTPConv'
    "BPConv": conv_network.BPConvNetwork,
    # NOTE: This one would reset the value to BPConv
    "BPConvCIFAR": conv_network.BPConvNetworkCIFAR,
    # args.network_type = 'BPConv'
}
# NOTE: These two entries are reusing the same classes, but with some different arguments.
# They aren't included in builders.py, they are instead done through pre-processing within main.py
meulemans_dtp_network_names_to_classes["DFA"] = with_fixed_constructor_arguments(
    # NOTE: use the previous entry, which corresponds to direct_feedback_networks.DDTPControlNetwork
    meulemans_dtp_network_names_to_classes["DMLPDTP2"],
    # freeze_fb_weights=True,
    size_hidden_fb=None,
    # size_mlp_fb=None,
    fb_activation="linear",
    # train_randomized=False,
)
meulemans_dtp_network_names_to_classes["DFAConv"] = with_fixed_constructor_arguments(
    meulemans_dtp_network_names_to_classes["DDTPConv"],
    # --- From main.py (when network_type == "DFAConv"):
    # args.freeze_fb_weights = True
    # args.network_type = 'DDTPConv'  # Overwrites the value.
    # args.fb_activation = 'linear'
    # args.train_randomized = False
    # --- From builders.py for network_type of "DDTPConv":
    # bias=not args.no_bias,
    # hidden_activation=args.hidden_activation,
    # feedback_activation=args.fb_activation,
    # initialization=args.initialization,
    # sigma=args.sigma,
    # plots=args.plots,
    # forward_requires_grad=forward_requires_grad,
    # nb_feedback_iterations=args.nb_feedback_iterations
    # --- Result:
    feedback_activation="linear",
)
meulemans_dtp_network_names_to_classes["DFAConvCIFAR"] = with_fixed_constructor_arguments(
    meulemans_dtp_network_names_to_classes["DDTPConvCIFAR"],
    # --- From main.py (when network_type == "DFAConvCIFAR"):
    # args.freeze_fb_weights = True
    # args.network_type = 'DDTPConvCIFAR'  # overwrites the value.
    # args.fb_activation = 'linear'
    # args.train_randomized = False
    # --- From builders.py for network_type of "DDTPConvCIFAR":
    # bias=not args.no_bias,
    # hidden_activation=args.hidden_activation,
    # feedback_activation=args.fb_activation,
    # initialization=args.initialization,
    # sigma=args.sigma,
    # plots=args.plots,
    # forward_requires_grad=forward_requires_grad,
    # nb_feedback_iterations=args.nb_feedback_iterations
    # --- Result:
    feedback_activation="linear",
)


# Maps from the name of a network in the paper, to the name of the network in the codebase.
# NOTE: Based on the README of the meulemans-dtp repo.
meulemans_paper_names_to_model_class: Dict[str, Callable[..., nn.Module]] = {
    "DTPDRL": meulemans_dtp_network_names_to_classes["DTPDR"],
    "DDTP-linear": with_fixed_constructor_arguments(
        meulemans_dtp_network_names_to_classes["DMLPDTP2"],
        fb_activation="linear",
        size_hidden_fb=None,  # NOTE: size_mlp_fp=None,
    ),
    "DDTP-RHL": meulemans_dtp_network_names_to_classes["DKDTP2"],
    "DTP": meulemans_dtp_network_names_to_classes["DTP"],
    "DTP-control": meulemans_dtp_network_names_to_classes["DDTPControl"],
    "BP": meulemans_dtp_network_names_to_classes["BP"],
    "DDTP-linear with CNN": meulemans_dtp_network_names_to_classes["DDTPConvCIFAR"],
    "DFA with CNN": meulemans_dtp_network_names_to_classes["DFAConvCIFAR"],
    "DDTP-control with CNN": meulemans_dtp_network_names_to_classes["DDTPConvControlCIFAR"],
    "BP with CNN": meulemans_dtp_network_names_to_classes["BPConvCIFAR"],
}

our_network_param_names_to_theirs = {
    LeNet: {
        "conv_0.conv.bias": "_layers.0._conv_layer.bias",
        "conv_0.conv.weight": "_layers.0._conv_layer.weight",
        "conv_1.conv.bias": "_layers.1._conv_layer.bias",
        "conv_1.conv.weight": "_layers.1._conv_layer.weight",
        "fc1.linear1.bias": "_layers.2._bias",
        "fc1.linear1.weight": "_layers.2._weights",
        "fc2.linear1.bias": "_layers.3._bias",
        "fc2.linear1.weight": "_layers.3._weights",
    }
}
their_network_param_names_to_ours = {
    model_type: {v: k for k, v in param_name_mapping.items()}
    for model_type, param_name_mapping in our_network_param_names_to_theirs.items()
}
