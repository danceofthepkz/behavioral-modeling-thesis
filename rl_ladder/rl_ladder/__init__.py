"""rl_ladder: a modular reinforcement-learning model ladder for mouse 2AFC learning."""

import os

# One numerical thread per process; parallelism comes from processes or Slurm.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

__version__ = "0.1.0"
