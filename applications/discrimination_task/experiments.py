from abc import ABC, abstractmethod

from bayesflow.amortizers import AmortizedPosterior, TwoLevelAmortizedPosterior
from bayesflow.networks import InvertibleNetwork, DeepSet, HierarchicalNetwork
from bayesflow.trainers import Trainer


class Experiment(ABC):
    """An interface for running a standardized simulated experiment."""

    @abstractmethod
    def __init__(self, model, *args, **kwargs):
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass


class StaticDiffusionExperiment(Experiment):
    pass


class StationaryDiffusionExperiment(Experiment):
    pass


class RandomWalkDiffusionExperiment(Experiment):
    pass


class RegimeSwitchingDiffusionExperiment(Experiment):
    pass

