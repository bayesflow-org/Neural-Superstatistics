from abc import ABC, abstractmethod


class DiffusionModel(ABC):
    """An interface for running a standardized simulated experiment."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def generate(self, batch_size, *args, **kwargs):
        pass
    
    @abstractmethod
    def configure(self, raw_dict, *args, **kwargs):
        pass


class StaticDiffusionModel(DiffusionModel):

    def __init__(self, *args, **kwargs):
        pass

    def generate(self, batch_size, *args, **kwargs):
        pass

    def configure(self, raw_dict):
        pass


class StationaryDiffusion(DiffusionModel):

    def __init__(self, *args, **kwargs):
        pass

    def generate(self, batch_size, *args, **kwargs):
        pass

    def configure(self, raw_dict):
        pass


class RandomWalkDiffusion(DiffusionModel):

    def __init__(self, *args, **kwargs):
        pass

    def generate(self, batch_size, *args, **kwargs):
        pass

    def configure(self, raw_dict):
        pass


class RegimeSwitchingDiffusion(DiffusionModel):

    def __init__(self, *args, **kwargs):
        pass

    def generate(self, batch_size, *args, **kwargs):
        pass

    def configure(self, *args, raw_dict):
        pass