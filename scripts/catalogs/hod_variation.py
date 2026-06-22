# scripts/catalogs/hod_variation.py
# generates a Sobol (quasi-random, low-discrepancy) HOD parameter variation and writes it
# to data/hod_vars/<name>.csv. mocks.py --variations loops over the csv rows,
# one mock (per seed) per variation. ranges + n_samples come from the hod_variation
# block in config.yaml; only the params listed there vary, the rest keep their baseline
# value from lrg_params.csv

# -----------
# IMPORTS
# -----------
import argparse
import copy
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import qmc

from src.utils import PATHS, load_config

def hod_var_path(variation_name):
    return PATHS.hod_var(variation_name)


## load a Sobol-sampled HOD variation csv. returns a list of (variation_id, overrides) where
## variation_id is the row index zero-padded to a fixed width (so dir names sort and
## map 1:1 to csv rows) and overrides maps each varied HOD param to its sampled value.
## params NOT in the csv keep their baseline value from lrg_params.csv.
def load_hod_var(csv_path):
    df = pd.read_csv(csv_path)
    width = len(str(len(df) - 1))
    param_cols = [c for c in df.columns if c != 'variation_id']
    return [(f'{int(row["variation_id"]):0{width}d}',
             {c: float(row[c]) for c in param_cols})
            for _, row in df.iterrows()]


# apply a set of HOD param overrides (one row of the Sobol-sampled variation) on top of the
# baseline config. overrides=None/empty -> baseline params unchanged (the base case).
def set_HOD_params(config, tracer='LRG', overrides=None):
    if not overrides:
        return config['HOD_params']
    HOD_params = copy.deepcopy(config['HOD_params'])
    HOD_params[f'{tracer}_params'].update(overrides)
    return HOD_params


@dataclass(frozen=True)
class HODParams:
    logM_cut: float
    logM1: float
    sigma: float
    kappa: float
    alpha: float
    alpha_c: float
    alpha_s: float

    Bcent: float = 0.
    Bsat: float = 0.

    s: float = 0.
    s_v: float = 0.
    s_p: float = 0.
    s_r: float = 0.

    Acent: float = 0.
    Asat: float = 0.

    ic: float = 1.


def load_hod_table(csv_path):
    df = pd.read_csv(csv_path)

    params = {}

    for _, row in df.iterrows():
        key = (
            row["TRACER"],
            row["MODEL"],
            float(row["ZSNAP"]),
        )

        params[key] = HODParams(
            logM_cut=row["logM_cut"],
            logM1=row["logM1"],
            sigma=row["sigma"],
            kappa=row["kappa"],
            alpha=row["alpha"],
            alpha_c=row["alpha_c"],
            alpha_s=row["alpha_s"],
            Bcent=row["Bcent"],
            Bsat=row["Bsat"],
        )

    return params


# load baseline hod params from lrg_params.csv for given tracer/model/z combination
# satellite profile params (s, s_v, s_p, s_r, Acent, Asat = 0) and incompleteness (ic = 1)
def load_lrg_hod_params(csv_path, tracer, model, z):
    hod_table = load_hod_table(csv_path)
    key = (tracer, model, float(z))
    if key not in hod_table:
        raise ValueError(f'no hod params found for tracer={tracer}, model={model}, z={z} in {csv_path}')
    return asdict(hod_table[key])

# -----------
# MAIN
# -----------
def main():
    parser = argparse.ArgumentParser(description='generate HOD parameter variation as csv with Sobol sampling')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite csv if it already exists')
    args = parser.parse_args()

    cfg      = load_config()
    sampling = cfg['hod_variation']
    name     = sampling['name']
    n_req    = int(sampling['n_samples'])
    seed     = int(sampling['seed'])
    ranges   = sampling['ranges']

    out_path = hod_var_path(name)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f'variation already exists at {out_path}; pass --overwrite to replace it')

    # Sobol balance properties hold at powers of 2; round up and warn if n_samples isn't one
    m = int(np.ceil(np.log2(n_req)))
    n = 2 ** m
    if n != n_req:
        print(f'rounding n_samples {n_req} -> {n} (2**{m}) for Sobol balance')

    params   = list(ranges.keys())          # column order = config order, fixed for reproducibility
    l_bounds = [ranges[p][0] for p in params]
    u_bounds = [ranges[p][1] for p in params]

    # draw 2**m points in the unit cube, then scale each column to its physical range
    sampler = qmc.Sobol(d=len(params), seed=seed)
    unit    = sampler.random_base2(m=m)
    scaled  = qmc.scale(unit, l_bounds, u_bounds)

    df = pd.DataFrame(scaled, columns=params)
    df.insert(0, 'variation_id', np.arange(n))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f'wrote {n} samples over {len(params)} params {params} to {out_path}')

if __name__ == '__main__':
    main()
