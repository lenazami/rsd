# scripts/gen/mocks.py
# generates galaxy mock catalogs w/ abacushod
# makes n mocks for fixed fiducial hods (tracer default: LRG).

# -----------
# IMPORTS
# -----------
import time
load_start = time.perf_counter()
import argparse
from dataclasses import replace
from pprint import pprint

import h5py

from rsd.utils import PATHS, MockID, SimID, load_config
from rsd.preprocess import cfg_combos, prep_sim, gen_cat

import_time = time.perf_counter() - load_start

# -----------
# MAIN
# -----------
parser = argparse.ArgumentParser(description='generate galaxy mock catalogs w/ abacushod')
parser.add_argument('-n', '--n-mocks', type=int, default=1,
                    help='number of mocks to generate for a given hod, default=1')
parser.add_argument('-t', '--tracer', type=str, default=None,
                    help='override the configured tracer: LRG, ELG, or QSO')
parser.add_argument('--hod-id', default=None,
                    help='explicit HOD id; overrides the tracer default (used by variations)')
args = parser.parse_args()
if args.n_mocks < 1:
    raise ValueError(f"n_mocks must be positive, not {args.n_mocks}")

print(f"Imports took {import_time:.3f}s")
script_start = time.perf_counter()
# load from config
cfg  = load_config()
combos = list(cfg_combos(cfg.get('sim_id', {})))

for over in combos:
    combo_start = time.perf_counter()
    sim = SimID.from_cfg(cfg, **over)
    mock = MockID.from_cfg(cfg, tracer=args.tracer, hod_id=args.hod_id, sim=sim)
    n_threads = cfg.get('n_threads', 32)
    requested = [replace(mock, seed=mock.seed + i) for i in range(args.n_mocks)]
    pending = []
    for seeded_mock in requested:
        row = seeded_mock.cat_lookup()
        if row is None:
            pending.append(seeded_mock)  # unregistered; registration mints the mock_id
            continue
        path = PATHS.mock_h5(seeded_mock)
        if not path.exists():
            pending.append(seeded_mock)  # registered but file gone; regenerate under same id
            continue
        with h5py.File(path, 'r') as f:
            missing = {'pos', 'vel'} - set(f)
            if missing:
                raise ValueError(f'invalid mock {path}: missing datasets {sorted(missing)}')
        print(f'{path.name} is already generated and registered; skipping')

    if not pending:
        continue

    # if not all subsamples exist, this will prep the sim, otherwise will just return
    prep_sim(sim, mock, cfg)
    
    sim_params = sim.sim_params()
    hod_params = mock.hod_params(sim.z)
    hod_pars = hod_params[mock.param_key]
    
    print('mocks using the following abacus sim params:')
    pprint(sim_params)
    print('using the following hod params:')
    pprint(hod_pars)
    print()

    init_start = time.perf_counter()
    print('initializing the HOD Ball object')
    from abacusnbody.hod.abacus_hod import AbacusHOD

    Ball = AbacusHOD(sim_params, hod_params)
    print(f'creating the HOD Ball took {time.perf_counter() - init_start:.3f}\n')

    # TODO: will eventually need to figure out a conditional for handling hod variations
    # mock gen loop
    mocks_start = time.time()
    for i, seeded_mock in enumerate(pending):
        print('-'*50)
        print(f'making mock {i+1} of {len(pending)} (seed={seeded_mock.seed})')
        path = gen_cat(Ball, seeded_mock, n_threads=n_threads)
        print(f'mock saved @ {path}')
    print('-'*50)
    print(f'made {len(pending)} mocks for {mock.tracer} h{mock.hod_id} in {time.time()-mocks_start:.3f}s')
    if len(combos)>1:
        print(f'combo took {time.perf_counter()-combo_start:.5f}s')
print(f'script ran successfully! {time.perf_counter()-script_start:.5f}s')
