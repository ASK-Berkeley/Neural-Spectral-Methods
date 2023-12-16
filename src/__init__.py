import os
import sys
import math

import operator as O
import itertools as I
import functools as F

# ---------------------------------------------------------------------------- #
#                                      JAX                                     #
# ---------------------------------------------------------------------------- #

import jax

import flax
import optax

import jax.numpy as np
import flax.linen as nn

# ---------------------------------------------------------------------------- #
#                                     TYPE                                     #
# ---------------------------------------------------------------------------- #

from abc import *
from typing import *

from jax import Array
from flax import struct

X = Union[Tuple["X", ...], List["X"], Array]
ϴ = Union[struct.PyTreeNode, "X", None]

Fx = Callable[..., "X"]      # real-valued function
Fϴ = Callable[..., "Fx"]    # parametrized function

# ---------------------------------------------------------------------------- #
#                                     CONST                                    #
# ---------------------------------------------------------------------------- #

e = np.e
π = np.pi

Δ = F.partial(np.einsum, "...ii -> ...")

# ---------------------------------------------------------------------------- #
#                                    RANDOM                                    #
# ---------------------------------------------------------------------------- #

from jax import random
RNG = Dict[str, random.KeyArray]

class RNGS(RNG):

    def __init__(self, prng: random.KeyArray, name: List[str]):
        keys = random.split(prng, len(name))
        super().__init__(zip(name, keys))

    def __next__(self) -> RNG:
        self.it = getattr(self, "it", 0) + 1
        return self.fold_in(self.it)

    def fold_in(self, key: Any) -> RNG:
        return { name: random.fold_in(data, hash(key))
             for name, data in self.items() }
