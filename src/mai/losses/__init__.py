from typing import Dict, Callable, Tuple

_LOSS_REG: Dict[str, Tuple[Callable, Callable]] = {}

def register_loss(name: str):
    def deco(fns:  Callable[[], Tuple[Callable, Callable]]):
        _LOSS_REG[name.lower()] = fns()
        return fns
    return deco

def build_loss(name: str) -> Tuple[Callable, Callable]:
    key = name.lower()
    if key not in _LOSS_REG:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(_LOSS_REG)}")
    return _LOSS_REG[key]

def available_losses() -> Dict[str, Tuple[Callable, Callable]]: return sorted(_LOSS_REG)
from mai.losses.cross_entropy import _cross_entropy