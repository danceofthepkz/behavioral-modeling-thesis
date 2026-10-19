"""Execution backends. Both run the same `tasks.run_unit`.

local  a process pool on this machine; units run in waves of model depth so
       that nested parents are fitted (and cached) before their children
slurm  one array index runs `chunk_size` consecutive units
       (see scripts/slurm/submit.sh)
"""

import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

from .service import read_json
from .tasks import run_unit


@lru_cache(maxsize=4)
def _plan(path):
    return read_json(path)


def _worker(plan_path, index):
    try:
        status, seconds = run_unit(_plan(str(plan_path)), index)
        return index, status, seconds, None
    except Exception as exc:  # recorded under <run>/failures/ by run_unit
        return index, "failed", 0.0, f"{type(exc).__name__}: {exc}"


def run_local(plan_path, n_jobs=1, quiet=False):
    plan = _plan(str(plan_path))
    units = plan["units"]
    waves = sorted({u["wave"] for u in units})
    counts = {"done": 0, "cached": 0, "failed": 0}
    failures = []
    tic = time.perf_counter()
    finished = 0
    for wave in waves:
        idx = [i for i, u in enumerate(units) if u["wave"] == wave]
        if n_jobs <= 1:
            results = (_worker(str(plan_path), i) for i in idx)
            pool = None
        else:
            pool = ProcessPoolExecutor(max_workers=n_jobs)
            futures = [pool.submit(_worker, str(plan_path), i) for i in idx]
            results = (f.result() for f in as_completed(futures))
        for i, status, seconds, error in results:
            finished += 1
            counts[status] += 1
            if error:
                failures.append((units[i]["id"], error))
            if not quiet:
                u = units[i]
                print(f"[{finished}/{len(units)}] {status:6s} {u['variant']} {u['task']} {u['id']}"
                      f"{f' ({seconds:.1f}s)' if status == 'done' else ''}"
                      f"{f'  ERROR {error}' if error else ''}", flush=True)
        if pool:
            pool.shutdown()
    elapsed = time.perf_counter() - tic
    print(f"Finished {len(units)} units in {elapsed:.1f}s: {counts}")
    for uid, error in failures[:20]:
        print(f"  failed {uid}: {error}")
    return counts


def run_chunk(plan_path, array_index, chunk_size):
    plan = _plan(str(plan_path))
    n = len(plan["units"])
    start, stop = array_index * chunk_size, min((array_index + 1) * chunk_size, n)
    if start >= n:
        print(f"Array index {array_index} has no units (n_units={n}, chunk={chunk_size})")
        return 0
    failed = 0
    for i in range(start, stop):
        _, status, seconds, error = _worker(str(plan_path), i)
        print(f"unit {i} {plan['units'][i]['id']}: {status} {seconds:.1f}s {error or ''}", flush=True)
        failed += status == "failed"
    return 1 if failed else 0


def n_array_tasks(plan_path, chunk_size):
    return math.ceil(_plan(str(plan_path))["n_units"] / chunk_size)
