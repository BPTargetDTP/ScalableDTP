import contextlib
import dataclasses
import io
import itertools
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Type, Union
import copy
import typing

import pandas as pd
import plotly.express as px
import torch
import tqdm
from pytorch_lightning import seed_everything
from target_prop.config import Config
from target_prop.metrics import compute_dist_angle
from target_prop.models import DTP
from target_prop.models.dtp import DTP, FeedbackOptimizerConfig, ForwardOptimizerConfig
from target_prop.networks import LeNet, Network
from target_prop.utils import make_reproducible
from target_prop.callbacks import get_dtp_grads
from torch import Tensor, nn
from torch.optim.optimizer import Optimizer
from target_prop.callbacks import get_backprop_grads
from torch.nn import functional as F

try:
    from meulemans_dtp.lib.train import (
        train_feedback_parameters,
        train_forward_parameters,
    )
except ImportError as e:
    raise RuntimeError(
        "You need the submodule of the meulemans DTP repo initialized to run this script. \n"
        "Run `git submodule init` and `git submodule update` and try again."
    )

from meulemans_dtp.main import Args
from meulemans_dtp.lib.conv_network import DDTPConvNetworkCIFAR
from meulemans_dtp.lib import builders, utils

from .utils import (
    _best_hparams,
    get_meulemans_args_for,
    translate,
    _check_bp_grads_are_identical,
    _check_outputs_are_identical,
    _check_forward_params_havent_moved,
    _get_forward_parameters,
    translate_back,
)


def figure_4p3(
    cache_file: Union[str, Path] = None,
    dataset: str = "cifar10",
    network_type: Type[Network] = LeNet,
    batch_size: int = 128,
    seeds: Sequence[int] = (123, 234, 345, 456, 567),
    beta: float = 0.001,
    n_pretraining_iterations: int = 10_000,
    dtp_hparams: DTP.HParams = None,
    network_hparams: Network.HParams = None,
):
    """
    Take randomly a batch of inputs X and associated ground-truth targets y.
    Take LeNet on Cifar-10 and compute BP gradients.
    Run the following 4 experiments:

    1. Leave Wb as such and compute DTP weight updates (for all the layers)
    2. Run DRL until convergence of Wb and compute DTP weight updates
    3. Same as 2) but with L-DRL
    4. Set Wb = W_f^T and compute DTP weight updates

    Compute the above quantities on 5 seeds (error bar on each bar) and for all the layers
    (FC and conv)
    """
    # Retrieve our best hyper-parameters for this dataset / model / architecture, use them as
    # defaults.
    best_dtp_hparams, best_network_hparams = _best_hparams[(dataset, DTP, network_type)]
    if dtp_hparams is None:
        assert isinstance(best_dtp_hparams, DTP.HParams)
        dtp_hparams = best_dtp_hparams
    if network_hparams is None:
        assert isinstance(best_network_hparams, network_type.HParams)
        network_hparams = best_network_hparams

    if cache_file is not None:
        cache_file = Path(cache_file)
    if cache_file and cache_file.exists():
        df = pd.read_hdf(cache_file)
    else:
        grouped_data = get_data_for_figure_4p3(
            dataset=dataset,
            network_type=network_type,
            seeds=seeds,
            batch_size=batch_size,
            beta=beta,
            n_pretraining_iterations=n_pretraining_iterations,
            dtp_hparams=dtp_hparams,
            network_hparams=network_hparams,
        )

        data = {}
        for (model, seed), (distances_dict, angles_dict) in grouped_data.items():
            assert set(distances_dict.keys()) == set(angles_dict.keys())
            for key, distance in distances_dict.items():
                angle = angles_dict[key]
                data[model, seed, key] = (distance, angle)

        indices = list(data.keys())
        values = [data[k] for k in indices]
        df = pd.DataFrame(
            values,
            index=pd.MultiIndex.from_tuples(indices, names=["model", "seed", "parameter"]),
            columns=pd.Index(["distance", "angle"], name="metric"),
        )
        if cache_file:
            df.to_hdf(cache_file, "w")

    gdf = df.groupby(level=("model", "parameter"), sort=True)
    df = pd.concat(
        [
            gdf.mean().rename(lambda c: f"{c}", axis="columns"),
            gdf.std().rename(lambda c: f"{c}_std", axis="columns"),
            gdf.count().rename(lambda c: f"{c}_count", axis="columns"),
        ],
        axis="columns",
    )

    # rename the names of the models:
    df = df.rename(
        {
            "DTP_symmetric": r"$\text{L-DRL}_{\text{sym}}$",
            "DTP_untrained": r"$\text{L-DRL}_{\text{init}}$",
            "DTP": r"$\text{L-DRL}$",
            "Meulemans-DTP": r"$\text{DRL}$",
            "Meulemans-DTP_untrained": r"$\text{DRL}_{\text{init}}$",
        },
    )
    # TODO: Don't include the bias terms.
    parameters = df.index.unique("parameter")
    bias_params = [p for p in parameters if "bias" in p]
    df = df.drop(labels=bias_params, level="parameter")

    df = df.reset_index()
    from plotly import graph_objects as go

    angles_fig: go.Bar = px.bar(
        df,
        x="model",
        y="angle",
        error_y="angle_std",
        barmode="group",
        color="parameter",
        title="Angle between DTP and Backprop Updates",
        # color_discrete_map={"cold": "blue", "warm": "orange", "hot": "red"},
        # points="all",
        width=1000,
        height=500,
    )
    angles_fig.update_layout(
        font_family="Serif",
        font_size=20,
        # font_color="blue",
        title_font_family="Serif",
        # title_font_color="red",
        # legend_title_font_color="green",
    )
    distances_fig = px.bar(
        df,
        x="model",
        y="distance",
        error_y="distance_std",
        barmode="group",
        color="parameter",
        title="Distances between DTP updates and Backprop Updates",
        # color_discrete_map={"cold": "blue", "warm": "orange", "hot": "red"},
        # points="all",
    )
    distances_fig.update_layout(
        font_family="Serif",
        font_size=20,
        # font_color="blue",
        title_font_family="Serif",
        # title_font_color="red",
        # legend_title_font_color="green",
    )
    return angles_fig, distances_fig


def get_data_for_figure_4p3(
    dataset: str,
    network_type: Type[Network],
    seeds: Sequence[int],
    dtp_hparams: DTP.HParams,
    network_hparams: Network.HParams,
    beta: float = 0.005,
    batch_size: int = 128,
    n_pretraining_iterations: int = 10_000,
) -> Dict[Tuple[str, int], Tuple[Dict[str, float], Dict[str, float]]]:
    # Number of training iterations to run to "train Wb until convergence"

    # Value of 'beta' to use in our DTP.
    # TODO: Do we need to set the same value in their codebase?

    config = Config(dataset=dataset, seed=42, device=torch.device("cuda"))
    seed_everything(42)
    dm = config.make_datamodule(batch_size=batch_size)
    dm.prepare_data()
    dm.setup()

    dataloader: DataLoader = dm.train_dataloader()  # type: ignore
    x: Tensor
    y: Tensor
    x, y = next(itertools.islice(dataloader, 1))
    x = x.cuda()
    y = y.cuda()
    # Network used for our DTP.

    # dict from model type to dict of distances and angles to the BackProp gradients.
    values: Dict[Tuple[str, int], Tuple[Dict[str, float], Dict[str, float]]] = {}
    for i, seed in enumerate(seeds):
        seed_everything(seed)
        # Create the DTP and Meulemans networks:
        dtp_network = network_type(
            in_channels=dm.dims[0], n_classes=dm.num_classes, hparams=network_hparams,
        )
        dtp_network = dtp_network.cuda()
        # NOTE: need to do a dummy forward pass, just to initialize all the weights.
        _ = dtp_network(x)

        # NOTE: This already does detach and clone.
        initial_network_weights = dtp_network.state_dict(keep_vars=False)

        with contextlib.redirect_stdout(io.StringIO()):
            args = get_meulemans_args_for(dataset=dataset, our_network_class=network_type)
        # NOTE: Setting this, just in case they use that value somewhere I haven't seen yet.
        args.random_seed = seed
        # NOTE: Setting this to False for the LeNet equivalent network to work.
        args.freeze_BPlayers = False
        # NOTE: Modifying these values so their architecture matches ours perfectly.
        args.hidden_activation = "elu"
        # NOTE: Setting beta to the same value as ours:
        args.target_stepsize = beta
        if isinstance(network_hparams, LeNet.HParams):
            args.no_bias = not network_hparams.bias

        meulemans_network = builders.build_network(args).cuda()
        meulemans_network.load_state_dict(translate(initial_network_weights), strict=False)

        assert isinstance(meulemans_network, DDTPConvNetworkCIFAR), "only works on this model atm."

        def reset_networks():
            # Reset the network weights to their initial states.
            # NOTE: This isn't really necessary, since the weights don't get updated.
            dtp_network.load_state_dict(initial_network_weights)
            meulemans_network.load_state_dict(translate(initial_network_weights), strict=False)

        reset_networks()
        _check_outputs_are_identical(our_network=dtp_network, meulemans_net=meulemans_network, x=x)

        # 1) Calculate the backprop gradients for our model and the Meulemans model.
        # NOTE: They are not the same atm because they use a different activation than we do.
        backprop_gradients = get_backprop_grads(dtp_network, x=x, y=y)
        meulemans_backprop_grads = get_meulemans_backprop_grads(
            meulemans_net=meulemans_network, x=x, y=y
        )

        _check_bp_grads_are_identical(
            our_network=dtp_network,
            meulemans_net=meulemans_network,
            our_backprop_grads=backprop_gradients,
            meulemans_backprop_grads=meulemans_backprop_grads,
        )

        # 2.1) Calculate the angle and distances between the grads from our DTP and their
        # corresponding backprop grads, when the feedback weights are randomly initialized.
        with make_reproducible(seed):
            values[("DTP_untrained", seed)] = get_dtp_grad_distances_and_angles(
                seed=seed,
                network=dtp_network,
                hparams=dtp_hparams,
                backprop_gradients=backprop_gradients,
                config=config,
                x=x,
                y=y,
                beta=beta,
                n_feedback_pretraining_iterations=0,
            )
        # Overkill, but just to be "safe".
        reset_networks()
        _check_outputs_are_identical(our_network=dtp_network, meulemans_net=meulemans_network, x=x)

        # 2.2) Calculate the angle and distances between the grads from Meuleman's DTP and their
        # corresponding backprop grads, when the feedback weights are randomly initialized.
        with make_reproducible(seed):
            # values[
            #     ("Meulemans-DTP_untrained", seed)
            # ] = get_meulemans_grad_distances_and_grad_angles(
            #     args=args,
            #     meulemans_net=meulemans_network,
            #     x=x,
            #     y=y,
            #     beta=beta,
            #     meulemans_backprop_grads=meulemans_backprop_grads,
            #     initial_network_weights=initial_network_weights,
            #     n_feedback_pretraining_iterations=n_pretraining_iterations,
            # )
            assert isinstance(network_hparams, LeNet.HParams)
            values[("Meulemans-DTP_untrained", seed)] = meulemans(
                network=dtp_network,
                network_hparams=network_hparams,
                x=x,
                y=y,
                beta=beta,
                backprop_gradients=backprop_gradients,
                seed=seed,
                n_pretraining_iterations=0,
            )

        reset_networks()
        _check_outputs_are_identical(our_network=dtp_network, meulemans_net=meulemans_network, x=x)

        # 3.1) Calculate the angle and distances between the grads from Meuleman's DTP and their
        # corresponding backprop grads, when the feedback weights are sufficiently trained.
        with make_reproducible(seed):
            # values[("Meulemans-DTP", seed)] = get_meulemans_grad_distances_and_grad_angles(
            #     args=args,
            #     meulemans_net=meulemans_network,
            #     x=x,
            #     y=y,
            #     meulemans_backprop_grads=meulemans_backprop_grads,
            #     initial_network_weights=initial_network_weights,
            #     n_feedback_pretraining_iterations=n_pretraining_iterations,
            #     beta=beta,
            # )
            values[("Meulemans-DTP", seed)] = meulemans(
                network=dtp_network,
                network_hparams=network_hparams,
                x=x,
                y=y,
                beta=beta,
                backprop_gradients=backprop_gradients,
                seed=seed,
                n_pretraining_iterations=n_pretraining_iterations,
            )

        # Overkill, but just to be "safe".
        reset_networks()
        _check_outputs_are_identical(our_network=dtp_network, meulemans_net=meulemans_network, x=x)

        # 3.2) Calculate the angle and distances between the grads from our DTP and their
        # corresponding backprop grads, when the feedback weights are sufficiently trained.
        with make_reproducible(seed):
            assert isinstance(dtp_hparams, DTP.HParams)
            values[("DTP", seed)] = get_dtp_grad_distances_and_angles(
                seed=seed,
                network=dtp_network,
                hparams=dtp_hparams,
                backprop_gradients=backprop_gradients,
                config=config,
                x=x,
                y=y,
                beta=beta,
                n_feedback_pretraining_iterations=n_pretraining_iterations,
            )

        # Overkill, but just to be "safe".
        reset_networks()

        # LAST: Need to init perpendicular weights now:
        # 4.2) Calculate the angle and distances between the grads from our DTP and backprop,
        # when the feedback weights are sufficiently trained.
        with make_reproducible(seed):
            assert isinstance(dtp_hparams, DTP.HParams)
            # We use the best hparams, but change this argument, which is used in
            # `DTP.create_backward_net`.
            # modified_hparams = dataclasses.replace(dtp_hparams, init_symetric_weights=True)
            # values[("DTP_symmetric", seed)] = get_dtp_grad_distances_and_angles(
            #     seed=seed,
            #     network=dtp_network,
            #     hparams=modified_hparams,
            #     backprop_gradients=backprop_gradients,
            #     config=config,
            #     x=x,
            #     y=y,
            #     beta=beta,
            #     n_feedback_pretraining_iterations=0,
            # )
            values[("DTP_symmetric", seed)] = get_dtp_sym_grad_distances_and_angles(
                seed=seed,
                network=dtp_network,
                hparams=dtp_hparams,
                backprop_gradients=backprop_gradients,
                config=config,
                x=x,
                y=y,
                beta=beta,
            )

    first_keys = list(values.values())[0][0].keys()
    for k, (distances, angles) in values.items():
        assert set(distances.keys()) == set(first_keys), k
        assert set(angles.keys()) == set(first_keys), k
    return values


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


def dtp_hparams():

    return DTP.HParams(
        feedback_training_iterations=[41, 51, 24],
        batch_size=256,
        noise=[0.41640228838517584, 0.3826261146623929, 0.1395382069358601],
        beta=0.4655,
        b_optim=FeedbackOptimizerConfig(
            type="sgd",
            lr=[0.0007188427494432325, 0.00012510321884615596, 0.03541466958291287],
            momentum=0.9,
        ),
        f_optim=ForwardOptimizerConfig(type="sgd", lr=0.03618, weight_decay=1e-4, momentum=0.9),
    )


def dtp_no_bias_model(dtp_hparams: DTP.HParams):
    config = Config(dataset="cifar10", num_workers=0, debug=False)
    datamodule = config.make_datamodule(batch_size=dtp_hparams.batch_size)
    network_hparams = LeNet.HParams(bias=False)
    network = LeNet(
        in_channels=datamodule.dims[0], n_classes=datamodule.num_classes, hparams=network_hparams
    )
    dtp_model = DTP(datamodule=datamodule, hparams=dtp_hparams, config=config, network=network)
    return dtp_model


def get_dtp_sym_grad_distances_and_angles(
    seed: int,
    network: Network,
    hparams: DTP.HParams,
    backprop_gradients: Dict[str, Tensor],
    config: Config,
    x: Tensor,
    y: Tensor,
    beta: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    # Fix seed

    # -- copied from `dtp_no_bias_model` --
    config = Config(dataset="cifar10", num_workers=0, debug=False, seed=seed)
    seed_everything(seed=seed, workers=True)

    datamodule = config.make_datamodule(batch_size=hparams.batch_size)
    # network_hparams = LeNet.HParams(bias=False)
    # network = LeNet(
    #     in_channels=datamodule.dims[0], n_classes=datamodule.num_classes, hparams=network_hparams
    # )
    hparams = dataclasses.replace(hparams, beta=beta)
    dtp_model = DTP(datamodule=datamodule, hparams=hparams, config=config, network=network)
    # return dtp_model
    ##

    # dtp_model = dtp_no_bias_model(hparams)  # rename for ease
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # device = torch.device("cpu")

    # Setup CIFAR10 datamodule with batch size 1
    # config = Config(dataset="cifar10", num_workers=0, debug=True)
    # datamodule = config.make_datamodule(batch_size=dtp_hparams.batch_size)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    # Get a batch
    # data, label = next(iter(datamodule.train_dataloader()))
    data = x.cuda()
    label = y.cuda()

    # Setup DTP model with symmetric weights
    from target_prop._weight_operations import init_symetric_weights

    init_symetric_weights(dtp_model.forward_net, dtp_model.backward_net)
    dtp_model.cuda()

    # Get backprop and DTP grads
    from target_prop.callbacks import get_backprop_grads, get_dtp_grads

    # Prove that this is the same:
    _backprop_grads = get_backprop_grads(dtp_model, data, label)
    for k, v in _backprop_grads.items():
        assert torch.allclose(backprop_gradients[k], v)
    # Use the ones provided above:
    backprop_grads = backprop_gradients

    dtp_grads = get_dtp_grads(dtp_model, data, label, temp_beta=dtp_model.hp.beta)

    # Compare gradients
    distances: Dict[str, float] = {}
    angles: Dict[str, float] = {}
    for (bp_param, bp_grad), (dtp_param, dtp_grad) in zip(
        backprop_grads.items(), dtp_grads.items()
    ):
        assert bp_param == dtp_param
        distance, angle = compute_dist_angle(bp_grad, dtp_grad)
        distances[bp_param] = distance
        angles[bp_param] = angle
    return distances, angles


def get_dtp_grad_distances_and_angles(
    seed: int,
    network: Network,
    hparams: DTP.HParams,
    backprop_gradients: Dict[str, Tensor],
    config: Config,
    x: Tensor,
    y: Tensor,
    beta: float,
    n_feedback_pretraining_iterations: int,
) -> Tuple[Dict[str, float], Dict[str, float]]:

    temp_config = dataclasses.replace(config, seed=seed, debug=True)
    # config = Config(dataset=dataset, seed=42, device=torch.device("cuda"))
    dm = temp_config.make_datamodule(batch_size=x.shape[0])
    # Replacing the number of feedback iterations per layer, setting it to 1 for all layers.
    hparams = dataclasses.replace(
        hparams,
        feedback_training_iterations=[1 for _ in hparams.feedback_training_iterations],
        beta=beta,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        model = DTP(datamodule=dm, network=network, hparams=hparams, config=temp_config)
    # Setup the forward optimizer:
    *feedback_optimizer_configs, forward_optimizer_config = model.configure_optimizers()
    forward_optimizer = forward_optimizer_config["optimizer"]
    assert isinstance(forward_optimizer, Optimizer)
    model._forward_optimizer = forward_optimizer

    # todo: Train the forward weights for `n_feedback_pretraining_iterations` iterations.
    if n_feedback_pretraining_iterations > 0:
        # NOTE: doing this so we can just set
        assert isinstance(model, DTP), "todo: Assuming this for now."
        feedback_optimizers = [config["optimizer"] for config in feedback_optimizer_configs]
        model._feedback_optimizers = feedback_optimizers + [None]

        progress_bar = tqdm.tqdm(
            range(n_feedback_pretraining_iterations),
            desc="Training Wb until convergence using our DTP",
        )

        for iteration in progress_bar:
            model.zero_grad(set_to_none=True)
            feedback_training_outputs = model.feedback_loss(x=x, y=y, phase="str")
            feedback_loss = feedback_training_outputs["loss"]
            if feedback_loss.requires_grad:
                raise RuntimeError(
                    "Assuming that we're using our DTP model, which should already have "
                    "backpropagated the feedback loss and updated the feedback weights, and thus "
                    "return a detached loss tensor without gradients."
                )
            progress_bar.set_postfix({"feedback loss": feedback_loss.item()})

        # trainer = Trainer(
        #     fast_dev_run=True,
        #     gpus=1,
        #     checkpoint_callback=False,
        #     limit_train_batches=1,
        #     limit_val_batches=1,
        #     limit_test_batches=1,
        # )
        # trainer.fit(model, datamodule=dm)

    dtp_grads = get_dtp_grads(model, x=x, y=y, temp_beta=beta)
    scaled_dtp_grads = {key: (1 / beta) * grad for key, grad in dtp_grads.items()}

    assert backprop_gradients.keys() == scaled_dtp_grads.keys()

    distances: Dict[str, float] = {}
    angles: Dict[str, float] = {}
    with torch.no_grad():
        # Same parameters should have gradients, regardless of if backprop or DTP is used.
        for parameter_name, backprop_gradient in backprop_gradients.items():
            scaled_dtp_grad = scaled_dtp_grads[parameter_name]
            distance, angle = compute_dist_angle(scaled_dtp_grad, backprop_gradient)
            distances[parameter_name] = distance
            angles[parameter_name] = angle
    return distances, angles


def get_meulemans_backprop_grads(
    meulemans_net: DDTPConvNetworkCIFAR, x: Tensor, y: Tensor,
) -> Dict[str, Tensor]:
    """ Returns the backprop gradients for the meulemans network. """
    # NOTE: Need to unfreeze the forward parameters of their network, since they apprear to be fixed

    meulemans_net_forward_params = _get_forward_parameters(meulemans_net)
    for name, forward_param in meulemans_net_forward_params.items():
        forward_param.requires_grad_(True)
    # NOTE: The forward pass through their network is a "regular" forward pass: Grads can flow
    # between all layers.
    predictions = meulemans_net(x)
    loss = F.cross_entropy(predictions, y)
    names = list(meulemans_net_forward_params.keys())
    parameters = [meulemans_net_forward_params[name] for name in names]
    meulemans_backprop_grads = dict(zip(names, torch.autograd.grad(loss, parameters)))
    return {k: v.detach() for k, v in meulemans_backprop_grads.items()}


def get_meulemans_grad_distances_and_grad_angles(
    args: Namespace,
    meulemans_net: DDTPConvNetworkCIFAR,
    x: Tensor,
    y: Tensor,
    meulemans_backprop_grads: Dict[str, Tensor],
    initial_network_weights: Dict[str, Tensor],
    beta: float,
    network_type: Type[Network] = LeNet,
    n_feedback_pretraining_iterations: int = 1000,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """ Returns
    - the Backprop gradients of their model
    - the distances from their updates to their backprop updates.
    - the angle between their updates and their backprop updates.
    """
    # Q: the lrs have to be the same between the different models?
    # TODO: The network I'm using for LeNet-equivalent doesn't actually allow this to work:
    # Says "frozen blabla isn't supported with OptimizerList"
    forward_optimizer, feedback_optimizer = utils.choose_optimizer(args, meulemans_net)

    # if isinstance(meulemans_net, DDTPMLPNetwork):
    #     x = x.flatten(1)

    if args.output_activation == "softmax":
        loss_function = nn.CrossEntropyLoss()
    else:
        assert args.output_activation == "sigmoid"
        loss_function = nn.MSELoss()

    meulemans_net.zero_grad(set_to_none=True)
    if n_feedback_pretraining_iterations > 0:
        for iteration in tqdm.tqdm(
            range(n_feedback_pretraining_iterations), desc="Running DRL until convergence of Wb",
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                # I think this is required every time, so the activations are updated
                predictions = meulemans_net.forward(x)
                train_feedback_parameters(args, meulemans_net, feedback_optimizer)

            # THis seems to be working for both the SimpleVGG and LeNet equivalents.

        # Double-check that the forward parameters have not been updated:
        _check_forward_params_havent_moved(
            meulemans_net, initial_network_weights=initial_network_weights
        )

    # --- Get the backprop gradients with their network:
    assert all(p.grad is None for name, p in _get_forward_parameters(meulemans_net).items())
    # NOTE: Recompute the predictions, since we already differentiated once.
    predictions = meulemans_net(x)
    assert all(p.grad is None for name, p in _get_forward_parameters(meulemans_net).items())

    # This propagates the targets backward, computes local forward losses, and sets the gradients
    # in the forward parameters' `grad` attribute.
    batch_accuracy, batch_loss = train_forward_parameters(
        args,
        meulemans_net,
        predictions,
        targets=y,
        loss_function=loss_function,
        forward_optimizer=forward_optimizer,
    )
    assert all(p.grad is not None for name, p in _get_forward_parameters(meulemans_net).items())

    # NOTE: the values in `p.grad` are the gradients from their DTP algorithm.
    meulemans_dtp_grads = {
        # NOTE: safe to ignore, from the check above.
        name: p.grad.detach()  # type: ignore
        for name, p in _get_forward_parameters(meulemans_net).items()
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
        translate_back(distances, network_type=network_type),
        translate_back(angles, network_type=network_type),
    )
