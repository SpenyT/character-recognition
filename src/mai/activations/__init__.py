from typing import Dict, Callable, Tuple

_ACTIV_REG: Dict[str, Tuple[Callable, Callable]] = {}

def register_activ(name: str):
    def deco(fns:  Callable[[], Tuple[Callable, Callable]]):
        activ_name = None if name is None else name.lower()
        _ACTIV_REG[activ_name] = fns()
        return fns
    return deco

def build_activ(name: str) -> Tuple[Callable, Callable]:
    key = None if name is None else name.lower()
    if key not in _ACTIV_REG:
        raise   
    return _ACTIV_REG[key]

def available_activations() -> Dict[str, Tuple[Callable, Callable]]: return sorted(_ACTIV_REG)
from mai.activations.default import _none
from mai.activations.relu import _relu
from mai.activations.sigmoid import _sigmoid
from mai.activations.tanh import _tanh