from abc import ABC
from dataclasses import dataclass
from typing import ClassVar
from simple_parsing.helpers.hparams.hyperparameters import HyperParameters
from target_prop.wandb_utils import LoggedToWandb

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


class Network(Protocol):
    @dataclass
    class HParams(HyperParameters, LoggedToWandb):
        # Where objects of this type can be parsed from in the wandb configs.
        _stored_at_key: ClassVar[str] = "net_hp"

    def __init__(self, in_channels: int, n_classes: int, hparams: HParams = None):
        ...
