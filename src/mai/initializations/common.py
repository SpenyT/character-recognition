from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from . import register_init

ArrayF = NDArray[np.floating]

# helper to validate and get fan_in and fan_out from shape
def _fan_in_fan_out(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) < 2: 
        fan_in = fan_out = int(np.sqrt(max(1, int(np.prod(shape))))) 
    else: 
        receptive_field = int(np.prod(shape[2:])) if len(shape) > 2 else 1 
        fan_in  = shape[0] * receptive_field if len(shape) > 2 else shape[0] 
        fan_out = shape[1] * receptive_field if len(shape) > 2 else shape[1] 
    return fan_in, fan_out


# Weight initializers
@register_init("xavier_uniform")
def _xavier_uniform(shape: tuple[int, ...], rng: np.random.Generator, dtype=np.float32) -> ArrayF:
    fan_in, fan_out = _fan_in_fan_out(shape)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape).astype(dtype)


@register_init("xavier_normal")
def _xavier_normal(shape: tuple[int, ...], rng: np.random.Generator, dtype=np.float32) -> ArrayF:
    fan_in, fan_out = _fan_in_fan_out(shape)
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return rng.normal(0.0, std, size=shape).astype(dtype)


@register_init("he_uniform")
def _he_uniform(shape: tuple[int, ...], rng: np.random.Generator, dtype=np.float32) -> ArrayF:
    fan_in, _ = _fan_in_fan_out(shape)
    limit = np.sqrt(6.0 / fan_in)
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)


@register_init("he_normal")
def _he_normal(shape: tuple[int, ...], rng: np.random.Generator, dtype=np.float32) -> ArrayF:
    fan_in, _ = _fan_in_fan_out(shape)
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, size=shape).astype(dtype)


@register_init("lecun_normal")
def _lecun_normal(shape: tuple[int, ...], rng: np.random.Generator, dtype=np.float32) -> ArrayF:
    fan_in, _ = _fan_in_fan_out(shape)
    std = np.sqrt(1.0 / fan_in)
    return rng.normal(0.0, std, size=shape).astype(dtype)


# Bias initializers
@register_init("zeroes")
def _zeroes(shape: tuple[int, ...], rng: np.random.Generator, dtype=np.float32) -> ArrayF:
    return np.zeros(shape, dtype=dtype)

@register_init("ones")
def _ones(shape: tuple[int, ...], rng: np.random.Generator, dtype=np.float32) -> ArrayF:
    return np.ones(shape, dtype=dtype)

