# scripts/01_prepare_sim.py
# preps AbacusSummit HDF5 particle subsamples needed by the HOD code.
# skips if subsamples already exist for given sim/redshift.

# -----------
# IMPORTS
# -----------
import yaml
from pathlib import Path

from src.utils import PROJECT_ROOT, load_config
from abacusnbody.hod.prepare_sim import main as prepare_main

# -----------
# MAIN
# -----------
def main():
    cfg      = load_config()
    sim_name = cfg['sim']['name']
    z_mock   = float(cfg['sim']['z'])
    tracer   = cfg['tracer']
    n_threads = cfg.get('n_threads', 32)

    sim_dir       = PROJECT_ROOT / 'abacus'
    subsample_dir = PROJECT_ROOT / 'subsamples/'
    subsample_path = subsample_dir / sim_name / f'z{z_mock:.3f}'

    if subsample_path.exists() and any(subsample_path.glob('*.h5')):
        print(f'subsamples already exist at {subsample_path}, nothing to do :)')
        return

    print(f'no subsamples found at {subsample_path}, running prepare_sim!')

    ## makes the directory for config yamls
    config_dir = PROJECT_ROOT / 'config' / 'sims'
    config_dir.mkdir(parents=True, exist_ok=True)

    prepare_cfg = {
        'sim_params': {
            'sim_name':      sim_name,
            'sim_dir':       str(sim_dir),
            'subsample_dir': str(subsample_dir),
            'output_dir':    str(PROJECT_ROOT / 'results' / 'mocks'),
            'z_mock':        z_mock,
            'cleaned_halos': 1,   ## remove unphysical CompaSO fragments
        },
        'HOD_params': {
            'tracer_flags': {tracer: True, 'ELG': False, 'QSO': False},
            'want_ranks':   True,    ## required for s_v satellite profile variation
            'want_AB':      False,
            'want_shear':   False,
            'want_rsd':     True,
        },
        'prepare_sim': {
            'Nthread_per_load': n_threads,
            'Nparallel_load':   1,
        },
    }

    cfg_path = config_dir / f'prepare_sim_{sim_name}_z{z_mock:.3f}.yaml'
    with open(cfg_path, 'w') as f:
        yaml.dump(prepare_cfg, f, default_flow_style=False)
    print(f'prepare_sim config written to {cfg_path}')

    prepare_main(str(cfg_path))
    print('prepare_sim done')

if __name__ == '__main__':
    main()
