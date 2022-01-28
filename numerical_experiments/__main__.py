"""Scripts to creates the figures of the paper.

Run this using:
```console
$ python -m numerical_experiments
```

NOTE: (@anonymous): It's a bit annoying, I agree. But it seems like the simplest option for now.
I might add command-line args for these later.
"""
from abc import abstractmethod
from simple_parsing import ArgumentParser, field
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Type
import wandb
import hashlib

from simple_parsing import choice, list_field, subparsers
from simple_parsing.helpers.serialization.serializable import Serializable
from target_prop.models.dtp import DTP, FeedbackOptimizerConfig, ForwardOptimizerConfig
from target_prop.networks.network import Network
from target_prop.networks.resnet import ResNet18, ResNet34
from .figure_4p2 import figure_4p2
from .figure_4p3 import figure_4p3
from .figure_extra import figure_extra
import matplotlib.pyplot as plt
from target_prop.networks import LeNet, SimpleVGG
from .utils import _best_hparams
import functools
import wandb
from wandb.sdk.wandb_run import Run


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(title="command", required=True)

    fig_42_parser = subparsers.add_parser("figure_4_2")
    fig_42_parser.add_arguments(Figure4p2, dest="fig")

    fig_43_parser = subparsers.add_parser("figure_4_3")
    fig_43_parser.add_arguments(Figure4p3, dest="fig")

    fig_43_parser = subparsers.add_parser("figure_4_3_extra")
    fig_43_parser.add_arguments(Figure4p3Extra, dest="fig")

    subparsers.metavar = "{" + ",".join(subparsers._name_parser_map.keys()) + "}"

    args = parser.parse_args()
    fig: PlotsConfig = args.fig
    fig.run()


@dataclass
class PlotsConfig(Serializable):
    dataset: str = choice("cifar10", "mnist", "fashion_mnist", default="cifar10")
    network_type: Type[Network] = choice({"lenet": LeNet, "simple_vgg": SimpleVGG}, default=LeNet)
    batch_size: int = 128
    n_pretraining_iterations: int = 1000
    seeds: List[int] = list_field(123, 234, 345, 456, 567)
    # NOTE: Trying to improve the alignment by changing some values here:

    # Parameters for the DTP algorithm used to create the plot.
    # When left unset, we will use the best hyper-parameters we have for the chosen
    # dataset / network type.
    dtp_hparams: Optional[DTP.HParams] = None

    # Parameters for the network architecture.
    # When left unset, we will use the best hyper-parameters we have for the chosen
    # dataset / network type.
    network_hparams: Optional[Network.HParams] = None

    use_wandb: bool = False

    def __post_init__(self):
        # Just announcing that these will be set in `setup_stuff`. (not necessary).
        self.wandb_run: Optional[Run] = None
        self.figures_dir: Path

        # Set the default values dynamically based on the choice of dataset / network type.
        # We use the best hyper-parameters as the default value, but this could also be changed.
        best_dtp_hparams, best_network_hparams = _best_hparams[self.dataset, DTP, self.network_type]
        if self.dtp_hparams is None:
            # If the parameters to use for DTP weren't passed, use the best values that we have.
            assert isinstance(best_dtp_hparams, DTP.HParams)
            self.dtp_hparams = best_dtp_hparams

        if self.network_hparams is None:
            assert isinstance(best_network_hparams, self.network_type.HParams)
            self.network_hparams = best_network_hparams

    def setup_stuff(self):
        """ Initialize the wandb stuff, create dirs for the figures, etc. """
        print(f"Plot options:")
        print(self.dumps_yaml(indent=1))
        if self.use_wandb:
            self.wandb_run: Run = wandb.init(
                project="scalingDTP", job_type="analysis", tags=["plots"]
            )
            self.wandb_run.config.update(self.to_dict())

        if self.wandb_run and self.wandb_run.name:
            self.figures_dir = Path(self.wandb_run.dir) / "figures"
            wandb.save(str(self.figures_dir))
        else:
            id = hashlib.md5(str(self.to_dict()).encode()).hexdigest()
            print(f"Unique id: {id}")
            self.figures_dir = Path("nice_figures") / id

        self.figures_dir.mkdir(exist_ok=True, parents=True)
        print(f"Figures will be saved locally in {self.figures_dir}")

    @abstractmethod
    def run(self):
        """ Creates the figure, save it locally and to wandb. """
        raise NotImplementedError()

    def save_plotly(self, fig, name: str):
        """ Save a given plotly figure, both locally and to wandb. """
        p = self.figures_dir / name
        fig.write_image(str(p))
        print(f"Figure saved locally at {p}")
        if self.use_wandb:
            wandb.save(str(p))


@dataclass
class Figure4p2(PlotsConfig):
    """ Options for creating figure 4.2 """

    # NOTE: Can only use LeNet atm, because other architectures don't have an equivalent in their
    # codebase.
    network_type: Type[Network] = choice({"lenet": LeNet}, default=LeNet)

    # Wether we should modify the Meulemans architecture to match ours (by using ELU activation and
    # removing padding in the maxpool layers) or if we should modify ours to match theirs (use tanh
    # and add padding in maxpool). NOTE: Only the first option is currently supported.
    modify_their_architecture: bool = True

    dtp_hparams: DTP.HParams = DTP.HParams(
        feedback_training_iterations=[41, 51, 24],
        batch_size=256,
        noise=[0.41640228838517584, 0.3826261146623929, 0.1395382069358601],
        # noise=[0.1],
        beta=0.4655,
        b_optim=FeedbackOptimizerConfig(
            type="sgd",
            lr=[0.0007188427494432325, 0.00012510321884615596, 0.03541466958291287],
            momentum=0.9,
        ),
        f_optim=ForwardOptimizerConfig(type="sgd", lr=0.03618, weight_decay=1e-4, momentum=0.9),
    )

    def __post_init__(self):
        super().__post_init__()

    def setup_stuff(self):
        super().setup_stuff()

    def run(self):
        self.setup_stuff()
        # TODO: Redo it using plotly, so that it's consistent with the other plots?
        fig = figure_4p2(
            dataset=self.dataset,
            batch_size=self.batch_size,
            n_pretraining_iterations=self.n_pretraining_iterations,
            seeds=self.seeds,
            network_type=self.network_type,
            modify_their_architecture=self.modify_their_architecture,
            dtp_hparams=self.dtp_hparams,
            network_hparams=self.network_hparams,
            cache_file=self.figures_dir / "fig_4_2_data.hdf5",
        )
        # TODO: make the figures pretty, convert them to matplotlib or something, add latex font.
        name = "figure_4_2.png"
        path = self.figures_dir / name
        fig.savefig(str(path), dpi=100)
        print(f"Saved figure locally at path {path}")
        plt.show(block=False)
        if self.use_wandb:
            img = wandb.Image(str(path))
            wandb.log({"Figure 4.2": img})
            wandb.save(str(path))


@dataclass
class Figure4p3(PlotsConfig):
    """ Options for creating figure 4.3 """

    dataset: str = choice("cifar10", default="cifar10")

    # NOTE: Can only use LeNet atm, because other architectures don't have an equivalent in their
    # codebase.
    network_type: Type[Network] = choice({"lenet": LeNet}, default=LeNet)

    dtp_hparams: DTP.HParams = DTP.HParams(
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

    # The value of `beta` to use, both in Meulemans and ours.
    beta: float = 1e-4

    network_hparams: LeNet.HParams = field(
        default_factory=functools.partial(LeNet.HParams, bias=False)
    )

    def __post_init__(self):
        super().__post_init__()

    def run(self):
        self.setup_stuff()
        angle_fig, distance_fig = figure_4p3(
            dataset=self.dataset,
            batch_size=self.batch_size,
            network_type=self.network_type,
            seeds=self.seeds,
            n_pretraining_iterations=self.n_pretraining_iterations,
            dtp_hparams=self.dtp_hparams,
            network_hparams=self.network_hparams,
            beta=self.beta,
            cache_file=self.figures_dir / "fig_4_3_data.hdf5",
        )

        # TODO: make the figures pretty, convert them to matplotlib or something, add latex font.
        # angle_fig.show()
        # distance_fig.show()
        self.save_plotly(angle_fig, "figure_4_3-angles.png")
        self.save_plotly(distance_fig, "figure_4_3-distances.png")
        if self.use_wandb:
            wandb.log(
                {"figure 4.3/angles": angle_fig, "figure 4.3/distances": distance_fig,}
            )


@dataclass
class Figure4p3Extra(PlotsConfig):
    """ Figure 4.3 "exta", Same as 4.3, but without the comparison to Meulemans DTP.
    This allows us to test all other architectures.
    """

    # Which dataset to use.
    dataset: str = choice("cifar10", "mnist", "fashion_mnist", default="cifar10")
    # The value of `beta` to use.
    beta: float = 1e-4

    # Network architecture to use. Compared to Figure 4.3, this has more options, and a different
    # default value.
    network_type: Type[Network] = choice(
        {"lenet": LeNet, "simple_vgg": SimpleVGG, "resnet18": ResNet18, "resnet34": ResNet34},
        default=SimpleVGG,
    )
    # NOTE: These are the params from LeNet, debugging with these atm.
    dtp_hparams: DTP.HParams = DTP.HParams(
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

    # The value of `beta` to use, both in Meulemans and ours.
    beta: float = 0.005

    network_hparams: LeNet.HParams = field(
        default_factory=functools.partial(LeNet.HParams, bias=False)
    )

    def run(self):
        self.setup_stuff()
        angle_fig, distance_fig = figure_extra(
            dataset=self.dataset,
            batch_size=self.batch_size,
            network_type=self.network_type,
            seeds=self.seeds,
            n_pretraining_iterations=self.n_pretraining_iterations,
            dtp_hparams=self.dtp_hparams,
            network_hparams=self.network_hparams,
            beta=self.beta,
            cache_file=Path(self.wandb_run.dir) / "fig_4_3_extra_data.hdf5",
        )
        angle_fig.show()
        distance_fig.show()
        self.save_plotly(angle_fig, "figure_4_3_extra-angles.png")
        self.save_plotly(distance_fig, "figure_4_3_extra-distances.png")
        if self.use_wandb:
            wandb.log(
                {
                    "figure 4.3 (extra)/angles": angle_fig,
                    "figure 4.3 (extra)/distances": distance_fig,
                }
            )


if __name__ == "__main__":
    main()
