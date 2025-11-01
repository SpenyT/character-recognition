from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, Protocol, List, Tuple, runtime_checkable
import numpy as np

class Layer(ABC):
    #is_training: bool = True
    @abstractmethod
    def forward_prop(self, x) -> np.ndarray:
        ...
    @abstractmethod
    def backward_prop(self, gradient_output: np.ndarray, learning_rate: float) -> np.ndarray:
        ...

class Trainable(Protocol):
    @property
    def params(self) -> Iterable[np.ndarray]: ...
    @property
    def grads(self) -> Iterable[np.ndarray]: ...

# Will be implementing this soon
# class TrainableLayer(Layer, Trainable, ABC):
#     def __init__(self) -> None:
#         self._params: List[np.ndarray] = []
#         self._grads: List[np.ndarray] = []

#     @property
#     def params(self) -> Tuple[np.ndarray, ...]:
#         return tuple(self._params)
    
#     @property
#     def grads(self) -> Tuple[np.ndarray, ...]:
#         return tuple(self._grads)

    