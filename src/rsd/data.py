import os
import warnings
import argparse
# import ast
import time
from dataclasses import replace
from pprint import pprint
import tempfile
import yaml

import numpy as np
import pandas as pd
import itertools
import h5py

from .utils import PATHS, COORDS, SimID, MockID

# redshifts w/ halo rvs
ALLOWED_ZS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0)

# ------------------
# Functions
# ------------------
# ----------- PREP SUBSAMPLES -----------
def slab_names(mock: MockID, i):
    # subsample slab filenames the HOD reader expects for slab i
    flags = mock.hod_flags()
    tf = flags['tracer_flags']
    mt    = '_MT' if (tf.get('ELG') or tf.get('QSO')) else ''
    ranks = '_withranks' if flags.get('want_ranks') else ''
    return (f'halos_xcom_{i}_seed600_abacushod_oldfenv{mt}_new.h5',
            f'particles_xcom_{i}_seed600_abacushod_oldfenv{mt}{ranks}_new.h5')

def check_subsamples(sim: SimID, mock: MockID, scratch=False):
    hinfo_dir = PATHS.abacus(sim, 'halo_info')
    n_slabs = len(sorted(hinfo_dir.glob('*.asdf')))
    if n_slabs == 0:
        raise FileNotFoundError(f'no halo_info .asdf files found in {hinfo_dir}')

    sub_dir = PATHS.subsample_dir(sim, scratch)
    missing = []
    for i in range(n_slabs):
        halo_f, part_f = (sub_dir / name for name in slab_names(mock, i))
        if not (halo_f.exists() and part_f.exists()):
            missing.append(i)

    return missing, n_slabs

def prep_sim(sim: SimID, mock: MockID, cfg, scratch=False):
    # generate the HDF5 particle subsamples for the HOD reader
    
    missing, n_slabs = check_subsamples(sim, mock, scratch)

    if not missing:
        print(f'all {n_slabs} slabs already prepared for {sim.realization} {sim.ztag}; nothing to do!')
        return
    if 0 < len(missing) < n_slabs:
        warnings.warn(f'incomplete generation for {sim.realization} {sim.ztag}! {len(missing)}/{n_slabs} slabs missing: {missing}. re-preparing...')

    print(f'preparing sim {sim.realization} {sim.ztag}...')
    
    from abacusnbody.hod.prepare_sim import main as prepare_main

    n_threads = cfg.get('n_threads', 32)
    n_parallel = cfg.get('n_parallel_load', 1)
    sim_params = sim.sim_params(scratch)

    # prepare_sim.main() concats subsample_dir + simname with no separator, so it
    # needs a trailing slash or it writes to e.g. 'data/subsamplesAbacusSummit_...'
    prepare_cfg = {
        'sim_params':  {**sim_params, 'cleaned_halos': 1,  # remove unphysical CompaSO fragments
                        'subsample_dir': os.path.join(sim_params['subsample_dir'], '')},
        'HOD_params':  mock.hod_flags(),
        'prepare_sim': {'Nthread_per_load': n_threads, 'Nparallel_load': n_parallel},
    }
    
    # abacushod requires a yaml file unless we start tampering with the modules themselves
    # which id prefer not to do, so we write temp files instead
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        yaml.dump(prepare_cfg, f, default_flow_style=False)
        cfg_path = f.name
    try:
        prepare_main(cfg_path, overwrite=0)
    finally:
        os.remove(cfg_path)

    missing, _ = check_subsamples(sim, mock, scratch)
    if missing:
        raise RuntimeError(f'prepare_sim finished with missing slabs: {missing}')
    print(f'done preparing sim: {sim.realization} {sim.ztag}')
    return prepare_cfg

# ----------- MOCKS -----------
def cfg_combos(section):
    # makes override dicts from cfg lists
    keys = list(section)
    vals = [v if isinstance(v, list) else [v] for v in section.values()]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def cat2h5(cat, mock, halo_data):
    cat = cat[mock.tracer]
    is_central = np.zeros(len(cat['x']), dtype='bool')
    is_central[:cat['Ncent']] = True

    # sort by (halo_id, centrals first), join halo centers on host halo id
    order = np.lexsort((~is_central, cat['id']))
    halo_id = cat['id'][order]
    # run_hod does not output host-halo info, so we add it ourselves
    pos = np.searchsorted(halo_data['hid'], np.asarray(halo_id))
    halo_pos, halo_vel = halo_data['hpos'][pos], halo_data['hvel'][pos]
    cols = {
        'pos':        np.column_stack([cat[c] for c in COORDS])[order],
        'vel':        np.column_stack([cat[f'v{c}'] for c in COORDS])[order],
        'mass':       cat['mass'][order],
        'is_central': is_central[order],
        'halo_id':    halo_id,
        'halo_pos':   halo_pos,
        'halo_vel':   halo_vel,
    }
    mock.register_cat()
    path = PATHS.mock_h5(mock)
    path.parent.mkdir(parents=True, exist_ok=True)  # auto-create data/mocks/ subdirs
    with h5py.File(path, 'w') as f:
        for k, v in cols.items():
            f.create_dataset(k, data=v)
    return path

# runs hod(seed) and writes cat to path; returns the path
def gen_cat(Ball, mock, n_threads=32):
    cat = Ball.run_hod(Ball.tracers, want_rsd=False, Nthread=n_threads,
                             reseed=mock.seed, verbose=True)
    return cat2h5(cat, mock, Ball.halo_data)

# ----------- MOCKS WITH HOD VARIATIONS -----------
# TODO: put hod_variation.py in here

def expand_range(field):
    field = str(field).strip().strip('{}')
    lo, _, hi = field.partition('-')
    if not lo.isdigit() or (hi and not hi.isdigit()):
        raise ValueError(f'bad catalog range {field!r}')
    return [str(i).zfill(len(lo)) for i in range(int(lo), int(hi or lo) + 1)]


def find_sims(suite, cosmos=None, z_range=(0.4, 1.4)):
    df = pd.read_csv(PATHS.data / 'simulations.csv', dtype=str, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    for column in df.columns:
        df[column] = df[column].str.strip()
    prefix = f'{suite}_c'
    rows = df[df['SimName'].str.startswith(prefix, na=False)]
    if rows.empty:
        raise ValueError(f'no {suite} rows in simulations.csv')

    zs = [z for z in ALLOWED_ZS if z_range[0] <= z <= z_range[1]]
    sims = []
    for _, row in rows.iterrows():
        selected = [
            cosmo for cosmo in expand_range(row['Cosm'])
            if cosmos is None or int(cosmo) in cosmos
        ]
        for cosmo, phase, z in itertools.product(
            selected, expand_range(row['Phase']), zs
        ):
            sim = SimID(name=suite.removeprefix('AbacusSummit_'), cosmo=cosmo, phase=phase, z=z)
            if not PATHS.abacus(sim, 'halo_info').is_dir():
                continue
            sims.append(sim)
    return sims


# ----------
# TRAINING
# ----------
def get_split_indices(length: int, test_size: float = 0.1, val_size: float = 0.05,
                      random_state: int = 42) -> tuple:
    """Generate train/val/test split indices."""
    from sklearn.model_selection import train_test_split

    indices = np.arange(length)
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size / (1.0 - test_size), random_state=random_state)
    return train_idx, val_idx, test_idx

