import copy
import dataclasses
import io
import itertools
from argparse import Namespace
from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple, Type, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import tqdm
from torch import Tensor

import contextlib
from pathlib import Path
from typing import Dict, Tuple, Type

from target_prop.config import Config
from target_prop.datasets import CIFAR10DataModule
from target_prop.layers import forward_all
from target_prop.metrics import compute_dist_angle
from target_prop.models import DTP
from target_prop.models.dtp import DTP
from target_prop.networks import LeNet, Network
from target_prop.utils import make_reproducible

import matplotlib
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes
import matplotlib
from matplotlib import pyplot as plt

try:
    from meulemans_dtp.lib import train
except ImportError as e:
    raise RuntimeError(
        "You need the submodule of the meulemans DTP repo initialized to run this script. \n"
        "Run `git submodule init` and `git submodule update` and try again."
    )
from meulemans_dtp.lib import builders, train, utils
from meulemans_dtp.lib.conv_network import DDTPConvNetworkCIFAR
from meulemans_dtp.lib.direct_feedback_networks import DDTPMLPNetwork

from .utils import _best_hparams, get_meulemans_args_for, translate


def figure_4p2(
    dataset: str = "cifar10",
    batch_size: int = 128,
    n_pretraining_iterations: int = 10_000,
    seeds: Sequence[int] = (1, 2, 3, 4, 5),
    network_type: Type[Network] = LeNet,
    modify_their_architecture: bool = True,
    cache_file: Union[str, Path] = None,
    dtp_hparams: DTP.HParams = None,
    network_hparams: Network.HParams = None,
):
    """

    1. Sample randomly Wf and Wb.
    2. Freeze Wf and pick a random batch of inputs.
    3. Run L-DRL (our DTP) and DRL (Meulemans) until convergence.

    Run this experiment with the LeNet architecture on MNIST or Cifar-10 and plot the curve for the
    output weights only.

    Run each curve on 5 seeds.
    """

    # Retrieve our best hyper-parameters for this dataset / model / architecture.
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
        df = get_data_for_figure_4p2(
            dataset=dataset,
            batch_size=batch_size,
            n_iterations=n_pretraining_iterations,
            seeds=seeds,
            network_type=network_type,
            modify_their_architecture=modify_their_architecture,
            dtp_hparams=dtp_hparams,
            network_hparams=network_hparams,
        )
        df.to_hdf(str(cache_file), "w")

    print(df)
    gdf = df.groupby(level=("model", "iteration"), sort=True)
    mean_df = gdf.mean()
    std_df = gdf.std()
    count_df = gdf.count()
    df = pd.concat(
        [
            gdf.mean().rename(lambda c: f"{c}", axis="columns"),
            gdf.std().rename(lambda c: f"{c}_std", axis="columns"),
            gdf.count().rename(lambda c: f"{c}_count", axis="columns"),
        ],
        axis="columns",
    )

    x = df.index.unique("iteration").to_numpy()
    model_names = df.index.unique("model")

    matplotlib.use("TkAgg")
    colors = {
        "L-DRL": "red",
        "DRL": "blue",
    }
    # if TYPE_CHECKING:
    # from matplotlib.axes._subplots import AxesSubplot

    font = {
        "size": 24,
        "family": "serif",
        # "sans-serif": ["Helvetica"],
        # "weight": "bold",
    }
    # family": "Helvetica", "font.size": 24
    matplotlib.rc("font", **font)
    fig: Figure
    axes: Sequence[Axes]
    fig, axes = plt.subplots(1, 2, sharex=True)
    err_width = 3
    for j, metric in enumerate(["angle", "distance"]):
        for i, model_name in enumerate(model_names):
            model_df = df.xs(model_name, level="model")
            y = model_df[metric]
            print(model_name, metric, y.min())
            import numpy as np

            y_std = model_df[f"{metric}_std"]
            axes[j].plot(x, y, label=model_name, color=colors[model_name])
            # annot_min(x=x, y=y, ax=axes[j])
            axes[j].fill_between(
                x,
                y - err_width * y_std,
                y + err_width * y_std,
                alpha=0.2,
                color=colors[model_name],
            )
            axes[j].grid(True, linewidth=1.5)
            # axes[j].set_yticks(range(0, 90, 5))
            # axes[j].set_yticklabels([str(v) if v % 10 == 0 else "" for v in range(0, 90, 5)])
            # axes[j].grid(True, which="major")
            if metric == "angle":
                axes[j].set_ylabel(r"$\angle(\theta^N, \omega^{N^\top})$", fontsize=26)
            else:
                axes[j].set_ylabel(r"$d(\theta^N, \omega^{n^\top})$")
        axes[j].legend(fontsize=20)
        axes[j].set_xlabel(r"$\omega^N$ training iterations")
        axes[0].set_xticks([0, 500, 1000])
        axes[0].set_yticks([0, 15, 30, 45, 60, 75, 90])
        # axes[j].set_title(f"{metric.capitalize()} between $W_b^T$ and $W_f$")
    matplotlib.rc("text", **{"usetex": True})
    fig.set_size_inches(12, 5)
    fig.tight_layout()
    return fig
    # Plotly equivalent:
    angle_fig: go.Scatter = px.line(
        df,
        x="iteration",
        y="angle_mean",
        error_y="angle_std",
        line_group="model",
        color="model",
        # color_discrete_map={"cold": "blue", "warm": "orange", "hot": "red"},
        # points="all",
        title="Angle between W_b^T and W_f in the last layer vs feedback training iteration",
    )
    distance_fig = px.line(
        df,
        x="iteration",
        y="distance_mean",
        error_y="distance_std",
        line_group="model",
        color="model",
        title="Distance between W_b^T and W_f in the last layer vs feedback training iteration",
        # color_discrete_map={"cold": "blue", "warm": "orange", "hot": "red"},
        # points="all",
    )
    return angle_fig, distance_fig


# def annot_min(x, y, ax=None, model_name: str):
#     xmin = x[np.argmin(y)]
#     xmin = x[-1]
#     ymin = y.min()
#     text = f"{model_name} min: {ymin:.2f}"
#     if not ax:
#         ax = plt.gca()
#     bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
#     # arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
#     kw = dict(
#         xycoords="data",
#         textcoords="axes fraction",
#         # arrowprops=arrowprops,
#         bbox=bbox_props,
#         ha="right",
#         va="top",
#         fontsize=20,
#     )
#     ax.annotate(text, xy=(xmin, ymin), xytext=(0.94, 0.50 - 0.1 * i), **kw)


def get_data_for_figure_4p2(
    dataset: str,
    network_type: Type[Network],
    n_iterations: int,
    seeds: Sequence[int],
    batch_size: int,
    dtp_hparams: DTP.HParams,
    network_hparams: Network.HParams,
    modify_their_architecture: bool,
) -> pd.DataFrame:
    config = Config(dataset=dataset, seed=42, num_workers=0)
    dm = config.make_datamodule(batch_size=batch_size)
    dm.prepare_data()
    dm.setup("fit")
    dataloader = dm.train_dataloader()
    xs, ys = zip(*itertools.islice(dataloader, len(seeds)))

    # model_function_dict = {
    #     "Meulemans-DTP": meulemans_fig_4p2,
    #     "DTP": functools.partial(dtp_fig_4p2, model_type=DTP),
    #     # "Vanilla-DTP": functools.partial(dtp_fig_4p2, model_type=VanillaDTP),
    #     # "Target-Prop": functools.partial(dtp_fig_4p2, model_type=TargetProp),
    # }

    # NOTE: We are replacing one entry in the hparams: Setting the number of iterations per batch
    # to 1 for all layers.
    dtp_hparams = dataclasses.replace(
        dtp_hparams,
        feedback_training_iterations=[1 for _ in dtp_hparams.feedback_training_iterations],
    )
    with contextlib.redirect_stdout(io.StringIO()):
        args = get_meulemans_args_for(dataset=dataset, our_network_class=network_type)
    # NOTE: Setting this, just in case they use that value somewhere I haven't seen yet.
    args.random_seed = 123
    # NOTE: Setting this to False for the LeNet equivalent network to work.
    args.freeze_BPlayers = False
    # NOTE: Setting beta to the same value as ours, even though this isn't really needed: only the
    # feedback weights are trained here, and beta is only used in forward weight training.
    args.target_stepsize = dtp_hparams.beta

    if modify_their_architecture:
        # NOTE: Modifying these values so their architecture matches ours perfectly.
        args.hidden_activation = "elu"
    else:
        raise NotImplementedError(
            f"Change out architecture to match theirs. (use tanh, set maxpool padding=1)"
        )

    data: Dict[Tuple[str, int, int], Tuple[float, float]] = {}

    def store_data(model_name: str, seed: int, distances: List[float], angles: List[float]):
        for i, (distance, angle) in enumerate(zip(distances, angles)):
            assert (model_name, seed, i) not in data
            data[(model_name, seed, i)] = (distance, angle)

    # Dict of initial parameters.
    # Will be overwritten for each seed.
    initial_parameters: Dict[str, Tensor] = {}

    for i, (seed, x, y) in enumerate(zip(seeds, xs, ys)):
        x = x.cuda()
        y = y.cuda()

        # Copy the values from the args, since their code modifies it.
        args_for_seed = copy.deepcopy(args)
        args_for_seed.random_seed
        config_for_seed = dataclasses.replace(config, seed=seed)

        with make_reproducible(seed):
            our_network = network_type(
                in_channels=dm.dims[0], n_classes=dm.num_classes, hparams=network_hparams,
            ).cuda()
            # Dummy forward pass to initialize the parameters.
            _ = our_network(x)

            if i > 0:
                assert initial_parameters
                # Debugging: Checking that the initializations are different between seeds.
                for name, parameter in our_network.named_parameters():
                    assert name in initial_parameters, initial_parameters.keys()
                    assert not torch.allclose(parameter, initial_parameters[name]), name
            initial_parameters = our_network.state_dict()

            meulemans_network = builders.build_network(args_for_seed).cuda()

            assert isinstance(meulemans_network, DDTPConvNetworkCIFAR), meulemans_network

            # Load the same forward weights for both.
            meulemans_network.load_state_dict(translate(initial_parameters), strict=False)

            with contextlib.redirect_stdout(io.StringIO()):
                model = DTP(
                    datamodule=dm,
                    network=our_network,
                    hparams=dtp_hparams,
                    config=config_for_seed,
                    network_hparams=our_network.hparams,
                )
                model.cuda()

            dtp_distances, dtp_angles = dtp_fig_4p2(
                model=model, x=x, y=y, n_iterations=n_iterations, datamodule=dm
            )
            store_data("L-DRL", seed, distances=dtp_distances, angles=dtp_angles)

            meulemans_distances, meulemans_angles = meulemans_fig_4p2(
                args=args_for_seed, model=meulemans_network, x=x, y=y, n_iterations=n_iterations,
            )
            store_data("DRL", seed, distances=meulemans_distances, angles=meulemans_angles)

        # for model_name, model_function in model_function_dict.items():
        #     print(model_name, model_function)
        #     with make_reproducible(seed):  # , redirect_stdout(io.StringIO()):
        #         distances, angles = model_function(
        #             seed=seed,
        #             x_batch=xs[0],
        #             y_batch=ys[0],
        #             dataset=dataset,
        #             network_type=network_type,
        #             n_iterations=n_iterations,
        #         )
        #         assert len(distances) == n_iterations, (len(distances), n_iterations)
        #         assert len(angles) == n_iterations, (len(angles), n_iterations)
        #     for i, (distance, angle) in enumerate(zip(distances, angles)):
        #         data[(model_name, seed, i)] = (distance, angle)
    indices = list(data.keys())
    values = [data[k] for k in indices]
    df = pd.DataFrame(
        values,
        index=pd.MultiIndex.from_tuples(indices, names=["model", "seed", "iteration"]),
        columns=pd.Index(["distance", "angle"], name="metric"),
    )
    return df


def meulemans_fig_4p2(
    args: Namespace, model: DDTPConvNetworkCIFAR, x: Tensor, y: Tensor, n_iterations: int = 1000,
) -> Tuple[List[float], List[float]]:
    model.cuda()

    # print(f"Param shapes for net of type {type(model)}:")
    # for param_name, param in model.named_parameters():
    #     print(f"{param_name}: {tuple(param.shape)}")

    # Q: the lrs have to be the same between the different models?
    _, feedback_optimizer = utils.choose_optimizer(args, model)

    # print(f"Sum of x (~id of x): {x.sum()=}, {y.sum()=}")
    x = x.cuda()

    if isinstance(model, DDTPMLPNetwork):
        x = x.flatten(1)

    distances, angles = [], []
    # TODO: Train the feedback weights until convergence. (I assume on that batch of data?)
    progress_bar = tqdm.tqdm(
        range(n_iterations), desc="Training feedback weights of Meulemans' DTP"
    )

    for iteration in progress_bar:
        with contextlib.redirect_stdout(io.StringIO()):
            # I think this is required every time, so the activations are updated
            predictions = model.forward(x)
            # This runs the entire loop of feedback weight training, and also performs the
            # optimizer step.
            train.train_feedback_parameters(args, model, feedback_optimizer)

        # TODO: Get the forward and feedback weights in a generic fashion.
        # THis seems to be working for both the SimpleVGG and LeNet equivalents.
        forward_weight = model._layers[-1]._weights.T
        feedback_weight = model._layers[-2]._fb_mlp.layers[0].weight  # type: ignore

        dist, angle = compute_dist_angle(forward_weight, feedback_weight)

        distances.append(dist)
        angles.append(angle)

        reconstruction_losses = [layer.reconstruction_loss for layer in model.layers]
        total_reconstruction_loss = sum(
            rec_loss for rec_loss in reconstruction_losses if rec_loss is not None
        )
        # print(f"Iteration {iteration}: {total_reconstruction_loss=}, {dist=}, {angle=}")
        progress_bar.set_postfix({"Reconstruction loss": total_reconstruction_loss})
    # print(f"{iteration}:  {dist=}, {angle=}")
    return distances, angles


def dtp_fig_4p2(
    model: DTP, x: Tensor, y: Tensor, n_iterations: int, datamodule: CIFAR10DataModule,
) -> Tuple[List[float], List[float]]:

    distances = []
    angles = []
    # NOTE: Need this to be true, so that it's a fair comparison with Meulemans.
    assert all(layer_iterations == 1 for layer_iterations in model.hp.feedback_training_iterations)

    start_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    # NOTE: Doing one batch of training to setup the optimizers.
    # Could possibly have unintended side-effects, but I'm not too worried about that.
    # trainer = Trainer(
    #     fast_dev_run=True,
    #     gpus=1,
    #     checkpoint_callback=False,
    #     limit_train_batches=1,
    #     limit_val_batches=1,
    #     limit_test_batches=1,
    # )
    # trainer.fit(model, datamodule=datamodule)
    *feedback_optimizer_configs, forward_optimizer_config = model.configure_optimizers()
    feedback_optimizers = [config["optimizer"] for config in feedback_optimizer_configs]
    forward_optimizer = forward_optimizer_config["optimizer"]
    model._feedback_optimizers = feedback_optimizers + [None]
    model._forward_optimizer = forward_optimizer

    # Restore the state of the model, just in case.
    model.load_state_dict(start_state_dict, strict=True)  # type: ignore

    x = x.cuda()
    y = y.cuda()
    model.cuda()

    # Train only the last layer.
    F_last = model.forward_net[-1]
    # note: they are aligned with the backward net.
    G_last = model.backward_net[0]
    G_optimizer = model.feedback_optimizers[0]
    assert G_optimizer is not None
    progress_bar = tqdm.tqdm(
        range(n_iterations), desc="Feedback weight training (forward weight fixed)"
    )
    for iteration in progress_bar:
        with torch.no_grad():
            activations = forward_all(model.forward_net, x, allow_grads_between_layers=False)
            x_i = activations[-2]
            y_i = activations[-1]
        noise_scale_i = model.feedback_noise_scales[0]
        loss = model.layer_feedback_loss(
            feedback_layer=G_last,
            forward_layer=F_last,
            input=x_i,
            output=y_i,
            noise_scale=noise_scale_i,
            noise_samples=model.hp.feedback_samples_per_iteration,
        )

        G_optimizer.zero_grad()
        loss.backward()
        G_optimizer.step()
        progress_bar.set_postfix({"loss": loss.item()})
        with torch.no_grad():
            metrics = compute_dist_angle(F_last, G_last)
            if isinstance(metrics, tuple):
                distance, angle = metrics
            else:
                if len(metrics) == 1:
                    distance, angle = metrics.popitem()[1]
                else:
                    nonzero_values = {k: v for k, v in metrics.items() if v != (0, 0)}
                    if len(nonzero_values) == 1:
                        distance, angle = nonzero_values.popitem()[1]
                    else:
                        # Expected to have only one distance/angle per block, but got this instead!
                        raise RuntimeError(metrics)
        distances.append(distance)
        angles.append(angle)

    return distances, angles
