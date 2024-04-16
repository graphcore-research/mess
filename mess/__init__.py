# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
import importlib.metadata

__version__ = importlib.metadata.version("mess")

from mess.basis import basisset
from mess.hamiltonian import Hamiltonian, minimise
from mess.structure import molecule

__all__ = ["molecule", "Hamiltonian", "minimise", "basisset"]


def setup_env():
    """Setup the environment for MESS

    This can be customised by setting the environment variables:

      MESS_ENABLE_FP64: enables float64 precision. Defaults to True.
      MESS_CACHE_DIR: JIT compilation cache location. Defaults to ~/.cache/mess
    """
    import os
    import os.path as osp

    from jax import config
    from jax.experimental.compilation_cache import compilation_cache as cc

    enable_fp64 = bool(os.environ.get("MESS_ENABLE_FP64", True))
    config.update("jax_enable_x64", enable_fp64)

    cache_dir = str(os.environ.get("MESS_CACHE_DIR", osp.expanduser("~/.cache/mess")))
    cc.set_cache_dir(cache_dir)


setup_env()
