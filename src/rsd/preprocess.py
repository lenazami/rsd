import os
import warnings
import argparse
# import ast
import json
import time
from dataclasses import replace
from pprint import pprint
import tempfile
import yaml

import numpy as np
import pandas as pd
import itertools
import h5py

from abacusnbody.metadata import get_meta

from .utils import PATHS, COORDS, SimID, MockID, register_sim



ALLOWED_ZS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0)


# ----------- MOCKS FROM SUBSAMPLES -----------
def cfg_combos(section):
    # makes override dicts from cfg lists
    keys = list(section)
    vals = [v if isinstance(v, list) else [v] for v in section.values()]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def match_halo_centers(halo_data, ids):
    # run_hod does not output host-halo info, so we add it ourselves
    pos = np.searchsorted(halo_data['hid'], np.asarray(ids))
    return halo_data['hpos'][pos], halo_data['hvel'][pos]

def setup_cosmology(sim):
    # mock info for the header
    m  = get_meta(sim.realization, redshift=sim.z)
    H0, z = m['H0'], sim.z
    Hz = H0 * m['HubbleNow']
    return {
        'sim_name':   sim.name,
        'boxsize':    float(m['BoxSize']),
        'z':          z,
        'H0':         H0,
        'Hz':         float(Hz),
        'rsd_factor': float((H0/100.) / (Hz/(1+z))),
    }
    
def cat2h5(mock_dict, mock, mock_info, hod_pars, halo_data):
    gal = mock_dict[mock.tracer]
    is_central = np.zeros(len(gal['x']), dtype='bool')
    is_central[:gal['Ncent']] = True

    # sort by (halo_id, centrals first), join halo centers on host halo id
    order = np.lexsort((~is_central, gal['id']))
    halo_id = gal['id'][order]
    halo_pos, halo_vel = match_halo_centers(halo_data, halo_id)
    cols = {
        'pos':        np.column_stack([gal[c] for c in COORDS])[order],
        'vel':        np.column_stack([gal[f'v{c}'] for c in COORDS])[order],
        'mass':       gal['mass'][order],
        'is_central': is_central[order],
        'halo_id':    halo_id,
        'halo_pos':   halo_pos,
        'halo_vel':   halo_vel,
    }

    register_sim(mock.sim)
    path = PATHS.mock_h5(mock)
    path.parent.mkdir(parents=True, exist_ok=True)  # auto-create data/mocks/ subdirs
    with h5py.File(path, 'w') as f:
        for k, v in cols.items():
            f.create_dataset(k, data=v)
        # sim/cosmo header + hod info & params
        f.attrs['MOCK_INFO'] = json.dumps(mock_info)
        f.attrs['HOD_INFO']  = json.dumps({
            'tracer':     mock.tracer,
            'hod_family': mock.hod_family,
            'hod_id':     mock.hod_id,
            'hod_model':  mock.hod_model,
            'seed':       mock.seed,
            'params':     hod_pars,
        })
    return path

# runs hod(seed) and writes cat to path; returns the path
def gen_cat(Ball, mock, mock_info, hod_pars, n_threads=32):
    mock_dict = Ball.run_hod(Ball.tracers, want_rsd=False, Nthread=n_threads,
                             reseed=mock.seed, verbose=True)
    return cat2h5(mock_dict, mock, mock_info, hod_pars, Ball.halo_data)

# # ----------- MOCKS WITH HOD VARIATIONS -----------
# # TODO: put hod_variation.py in here

# def expand_range(field):
#     # catalog range fields: "001-005" -> ["001", ..., "005"]; "000" -> ["000"]
#     lo, _, hi = str(field).strip().partition('-')
#     assert lo.isdigit() and (hi == '' or hi.isdigit()), f'bad catalog range {field!r}'
#     return [str(i).zfill(len(lo)) for i in range(int(lo), int(hi or lo) + 1)]

# def find_sims(name='base', cosmos=None, z_range=(0.4, 1.4)):
#     # enumerate realizations from the AbacusSummit catalog (data/simulations.csv)
#     df = pd.read_csv(PATHS.data / 'simulations.csv', dtype=str, skipinitialspace=True)
#     df.columns = df.columns.str.strip()
#     rows = df[df['SimName'].str.startswith(f'AbacusSummit_{name}_c')]
#     assert len(rows), f'no AbacusSummit_{name} rows in simulations.csv'

#     zs = [z for z in ALLOWED_ZS if z_range[0] <= z <= z_range[1]]
#     sims = []
#     for _, row in rows.iterrows():
#         keep = [c for c in expand_range(row['Cosm']) if cosmos is None or int(c) in cosmos]
#         for cosmo, phase, z in itertools.product(keep, expand_range(row['Phase']), zs):
#             sim = SimID(name=name, z=z, cosmo=cosmo, phase=phase)
#             assert PATHS.abacus(sim, 'halo_info').is_dir(), \
#                 f'catalog sim missing on disk: {PATHS.abacus(sim, "halo_info")}'
#             sims.append(sim)
#     return sims


# ----------- PREP SUBSAMPLES FROM ABACUS HD5 -----------

def slab_names(mock: MockID, i):
    # subsample slab filenames the HOD reader expects for slab i
    flags = mock.hod_flags()
    tf = flags['tracer_flags']
    assert mock.tracer in tf, f'bad tracer flags for {mock.tracer}'
    mt    = '_MT' if (tf.get('ELG') or tf.get('QSO')) else ''
    ranks = '_withranks' if flags.get('want_ranks') else ''
    return (f'halos_xcom_{i}_seed600_abacushod_oldfenv{mt}_new.h5',
            f'particles_xcom_{i}_seed600_abacushod_oldfenv{mt}{ranks}_new.h5')

def check_subsamples(sim: SimID, mock: MockID, scratch=False):
    # Count the slab outputs expected by the HOD reader.
    # slabs the HOD will try to read = number of downloaded halo_info files
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

def prep_sim(sim: SimID, mock: MockID, cfg, prep_force: bool, scratch=False):
    # generate the HDF5 particle subsamples for the HOD reader
    
    missing, n_slabs = check_subsamples(sim, mock, scratch)

    if not missing and not prep_force:
        print(f'all {n_slabs} slabs already prepared for {sim.realization} {sim.ztag}; nothing to do!')
        return
    if 0 < len(missing) < n_slabs:
        warnings.warn(f'incomplete generation for {sim.realization} {sim.ztag}! {len(missing)}/{n_slabs} slabs missing: {missing}. re-preparing...')

    print(f'preparing sim {sim.realization} {sim.ztag}...')
    
    from abacusnbody.hod.prepare_sim import main as prepare_main

    n_threads = cfg.get('n_threads', 32)
    # 
    # each parallel load is a separate process holding a full slab (~16GB for base),
    # so this is the memory<->wall knob: P slabs at once = P x per-slab RAM
    n_parallel = cfg.get('n_parallel_load', 1)
    sim_params = sim.sim_params(scratch)

    # prepare_sim.main() concatenates subsample_dir + simname with no separator, so it
    # needs a trailing slash or it writes to e.g. 'data/subsamplesAbacusSummit_...'
    prepare_cfg = {
        'sim_params':  {**sim_params, 'cleaned_halos': 1,  # remove unphysical CompaSO fragments
                        'subsample_dir': os.path.join(sim_params['subsample_dir'], '')},
        'HOD_params':  mock.hod_flags(),
        'prepare_sim': {'Nthread_per_load': n_threads, 'Nparallel_load': n_parallel},
    }
    
    # abacushod requires a yaml file unless we start tampering with the 
    # modules themselves which id prefer not to do, so we write temp files instead
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        yaml.dump(prepare_cfg, f, default_flow_style=False)
        cfg_path = f.name
    try:
        prepare_main(cfg_path, overwrite=int(prep_force))
    finally:
        os.remove(cfg_path)

    missing, _ = check_subsamples(sim, mock, scratch)
    if missing:
        raise RuntimeError(f'prepare_sim finished with missing slabs: {missing}')
    print('done preparing sim.')
    print(f'all {n_slabs} slabs prepared for {sim.realization} {sim.ztag}')
    return prepare_cfg
