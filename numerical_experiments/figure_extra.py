import dataclasses
from pathlib import Path
from typing import (
    Dict,
    List,
    Sequence,
    Tuple,
    Type,
    Union,
)

import pandas as pd
import plotly.express as px
import torch
import itertools
from pytorch_lightning import seed_everything
from target_prop.callbacks import get_backprop_grads
from target_prop.config import Config
from target_prop.models import DTP
from target_prop.models.dtp import DTP
from target_prop.networks import Network, SimpleVGG
from target_prop.utils import make_reproducible
from torch import Tensor

from .utils import _best_hparams
from .figure_4p3 import get_dtp_grad_distances_and_angles


def figure_extra(
    dataset: str = "cifar10",
    network_type: Type[Network] = SimpleVGG,
    batch_size: int = 32,
    seeds: Sequence[int] = (123, 234, 345, 456, 567),
    beta: float = 0.005,
    n_pretraining_iterations: int = 10_000,
    dtp_hparams: DTP.HParams = None,
    network_hparams: Network.HParams = None,
    cache_file: Union[str, Path] = None,
):
    """
    Same as 4.3, but only comparing between different initial conditions for our own netorks
    (no Meulemans).
    This is more flexible than figure 4.3, where we can only currently use LeNet, since we don't
    have an equivalent in their codebase for our other networks.
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
        df = pd.read_hdf(str(cache_file))
    else:
        grouped_data = get_data_for_extra_figure(
            dataset=dataset,
            network_type=network_type,
            seeds=list(seeds),
            batch_size=batch_size,
            beta=beta,
            dtp_hparams=dtp_hparams,
            network_hparams=network_hparams,
            n_pretraining_iterations=n_pretraining_iterations,
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
            df.to_hdf(str(cache_file), "w")

    gdf = df.groupby(level=("model", "parameter"), sort=True)
    df = pd.concat(
        [
            gdf.mean().rename(lambda c: f"{c}", axis="columns"),
            gdf.std().rename(lambda c: f"{c}_std", axis="columns"),
            gdf.count().rename(lambda c: f"{c}_count", axis="columns"),
        ],
        axis="columns",
    )

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
    angles_fig = px.bar(
        df,
        x="model",
        y="angle",
        error_y="angle_std",
        barmode="group",
        color="parameter",
        title="Angle between DTP updates and Backprop Updates",
        # color_discrete_map={"cold": "blue", "warm": "orange", "hot": "red"},
        # points="all",
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


def get_data_for_extra_figure(
    dataset: str,
    seeds: List[int],
    beta: float,
    batch_size: int,
    network_type: Type[Network],
    n_pretraining_iterations: int,
    dtp_hparams: DTP.HParams,
    network_hparams: Network.HParams,
) -> Dict[Tuple[str, int], Tuple[Dict[str, float], Dict[str, float]]]:
    # assert network_type is SimpleVGG
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

        # Creates the meulemans equivalent network and also initialize the weights to the same
        # values

        def reset_network():
            # Reset the network weights to their initial states.
            # NOTE: This isn't really necessary, since the weights don't get updated.
            dtp_network.load_state_dict(initial_network_weights)

        reset_network()

        # 1) Calculate the backprop gradients for our model and the Meulemans model.
        # NOTE: They are not the same atm because they use a different activation than we do.
        backprop_gradients = get_backprop_grads(dtp_network, x=x, y=y)

        # 2.1) Calculate the angle and distances between the grads from our DTP and their
        # corresponding backprop grads, when the feedback weights are randomly initialized.
        with make_reproducible(seed):
            assert isinstance(dtp_hparams, DTP.HParams)
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
        reset_network()

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
        reset_network()

        # LAST: Need to init perpendicular weights now:
        # 4.2) Calculate the angle and distances between the grads from our DTP and backprop,
        # when the feedback weights are sufficiently trained.
        with make_reproducible(seed):
            assert isinstance(dtp_hparams, DTP.HParams)
            # We use the best hparams, but change this argument, which is used in
            # `DTP.create_backward_net`.
            modified_hparams = dataclasses.replace(dtp_hparams, init_symetric_weights=True)
            values[("DTP_symmetric", seed)] = get_dtp_grad_distances_and_angles(
                seed=seed,
                network=dtp_network,
                hparams=modified_hparams,
                backprop_gradients=backprop_gradients,
                config=config,
                x=x,
                y=y,
                beta=beta,
                n_feedback_pretraining_iterations=0,
            )

    first_keys = list(values.values())[0][0].keys()
    for k, (distances, angles) in values.items():
        assert set(distances.keys()) == set(first_keys), k
        assert set(angles.keys()) == set(first_keys), k
    return values
