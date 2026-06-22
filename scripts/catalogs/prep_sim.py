# scripts/catalogs/prep_sim.py
# preps the HDF5 particle subsamples the HOD code needs; resumable (overwrite=0).
# slab count = number of downloaded halo_info_*.asdf files (contiguous from 000).

# -----------
# IMPORTS
# -----------
import os
from pathlib import Path

import yaml

from src.utils import PATHS, load_config, build_sim_params, build_hod_flags

# -----------
# FUNCTIONS
# -----------

# counts # of slabs under data/subsamples, 
# checks for contiguous ids: 0..n_slabs-1 in case of partial generations
def check_subsamples(sim_params, hod_flags):
    sim_name = sim_params['sim_name']
    z = sim_params['z_mock']

    # slabs the HOD will try to read = number of downloaded halo_info files
    halo_info_dir = (Path(sim_params['sim_dir']) / sim_name / 'halos'
                     / f'z{z:.3f}' / 'halo_info')
    n_slabs = len(sorted(halo_info_dir.glob('*.asdf')))
    if n_slabs == 0:
        raise FileNotFoundError(f'no halo_info .asdf files found in {halo_info_dir}')

    subsample_dir = Path(sim_params['subsample_dir']) / sim_name / f'z{z:.3f}'

    tf = hod_flags['tracer_flags']
    mt    = '_MT' if (tf.get('ELG') or tf.get('QSO')) else ''
    ranks = '_withranks' if hod_flags.get('want_ranks') else ''

    missing = []
    for i in range(n_slabs):
        halo_f = subsample_dir / f'halos_xcom_{i}_seed600_abacushod_oldfenv{mt}_new.h5'
        part_f = subsample_dir / f'particles_xcom_{i}_seed600_abacushod_oldfenv{mt}{ranks}_new.h5'
        if not (halo_f.exists() and part_f.exists()):
            missing.append(i)

    return len(missing) == 0, missing, n_slabs

# -----------
# MAIN
# -----------
def main():
    cfg      = load_config()
    sim_name = cfg['sim']['name']
    z_mock   = float(cfg['sim']['z'])
    tracer   = cfg['tracer']
    n_threads = cfg.get('n_threads', 32)

    sim_params = build_sim_params(sim_name, z_mock)
    hod_flags = build_hod_flags(tracer, want_rsd=True)

    complete, missing, n_slabs = check_subsamples(sim_params, hod_flags)
    if complete:
        print(f'all {n_slabs} slabs already prepared for {sim_name} z{z_mock:.3f}; nothing to do')
        return
    print(f'{len(missing)} of {n_slabs} slabs missing {missing}; running prepare_sim '
          f'(overwrite=0, so already-prepared slabs are skipped)')

    from abacusnbody.hod.prepare_sim import main as prepare_main

    # prepare_sim.main() concatenates subsample_dir + simname with no separator, so it
    # needs a trailing slash or it writes to e.g. 'data/subsamplesAbacusSummit_...'
    prepare_cfg = {
        'sim_params':  {**sim_params, 'cleaned_halos': 1,  # remove unphysical CompaSO fragments
                        'subsample_dir': os.path.join(sim_params['subsample_dir'], '')},
        'HOD_params':  hod_flags,
        'prepare_sim': {'Nthread_per_load': n_threads, 'Nparallel_load': 1},
    }
    cfg_path = PATHS.prepare_sim_config(sim_name, z_mock)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, 'w') as f:
        yaml.dump(prepare_cfg, f, default_flow_style=False)
    print(f'prepare_sim config written to {cfg_path}')

    prepare_main(str(cfg_path), overwrite=0)
    print('prepare_sim done')

    complete, missing, n_slabs = check_subsamples(sim_params, hod_flags)
    if not complete:
        raise RuntimeError(f'prepare_sim finished but slabs still missing {missing} '
                           f'of {n_slabs}; check the prepare_sim output above')
    print(f'all {n_slabs} slabs prepared for {sim_name} z{z_mock:.3f}')

if __name__ == '__main__':
    main()
