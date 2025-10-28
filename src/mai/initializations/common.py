# src/mai/initializers/common.py
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from . import register_initializer

ArrayF = NDArray[np.floating]

# helper to validate and get fan_in and fan_out from shape
def _fan_in_fan_out(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) < 2:
        raise ValueError(f"Expected (fan_in, fan_out) shape, got {shape}")
    return shape[0], shape[1]


# Weight initializers
@register_initializer("xavier_uniform")
def _xavier_uniform(shape: tuple[int, ...]) -> ArrayF:
    fan_in, fan_out = _fan_in_fan_out(shape)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=shape).astype(np.float32)

@register_initializer("xavier_normal")
def _xavier_normal(shape: tuple[int, ...]) -> ArrayF:
    fan_in, fan_out = _fan_in_fan_out(shape)
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return (np.random.randn(*shape) * std).astype(np.float32)

@register_initializer("he_uniform")
def _he_uniform(shape: tuple[int, ...]) -> ArrayF:
    fan_in, _ = _fan_in_fan_out(shape)
    limit = np.sqrt(6.0 / fan_in)
    return np.random.uniform(-limit, limit, size=shape).astype(np.float32)

@register_initializer("he_normal")
def _he_normal(shape: tuple[int, ...]) -> ArrayF:
    fan_in, _ = _fan_in_fan_out(shape)
    std = np.sqrt(2.0 / fan_in)
    return (np.random.randn(*shape) * std).astype(np.float32)

@register_initializer("lecun_normal")
def _lecun_normal(shape: tuple[int, ...]) -> ArrayF:
    fan_in, _ = _fan_in_fan_out(shape)
    std = np.sqrt(1.0 / fan_in)
    return (np.random.randn(*shape) * std).astype(np.float32)


# Bias initializers
@register_initializer("zeroes")
def _zeroes(shape: tuple[int, ...]) -> ArrayF:
    return np.zeros(shape, dtype=np.float32)

@register_initializer("ones")
def _ones(shape: tuple[int, ...]) -> ArrayF:
    return np.ones(shape, dtype=np.float32)

