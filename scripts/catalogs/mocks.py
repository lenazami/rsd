# scripts/02_generate_mocks.py
# gens galaxy mock catalogs w/ AbacusHOD.
# allows for HOD parameter variations defined in config.yaml.
# use --test to run a single base case!

# -----------
# IMPORTS
# -----------
import argparse
import ast
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path
from pprint import pprint

import numpy as np
import yaml
from astropy.table import Table

import camb
from camb.dark_energy import DarkEnergyPPF
from abacusnbody.hod.abacus_hod import AbacusHOD

from src.utils import (PROJECT_ROOT, HOD_CSV, load_config, get_param_dict,
                       load_hod_table, mock_path)


# -----------
# FUNCTIONS
# -----------

## this function parses through the abacus.pars file
def get_abacus_pars(filename):

    abacus_pars = {}
    with open(filename) as f:
        for line in f.readlines()[:-2]:
            key, val = (x.strip() for x in line.strip().split('=', 1))
            try:
                abacus_pars[key] = ast.literal_eval(val)
            except Exception:
                abacus_pars[key] = val
    return abacus_pars

## this function scrapes the abacus_pars to get CAMB cosmology params
def abacus_pars2camb_pars(abacus_pars):
    cosmo_info = {
        'H0':        abacus_pars['H0'],
        'ombh2':     abacus_pars['omega_b'],
        'omch2':     abacus_pars['omega_cdm'],
        'omk':       abacus_pars['Omega_K'],
        'tau':       0.06,
        'As':        2e-9,
        'ns':        abacus_pars['n_s'],
        'thetastar': 0.0104109,   # Planck 2018 theta_*, fixes CMB sound horizon
                                  # this allows for computing H0 using CAMB while keeping SH fixed
        'w0': abacus_pars['w0'],
        'wa': abacus_pars['wa'],
    }
    camb_pars = camb.set_params(
        ombh2=cosmo_info['ombh2'], omch2=cosmo_info['omch2'], omk=cosmo_info['omk'],
        tau=cosmo_info['tau'], As=cosmo_info['As'], ns=cosmo_info['ns'],
        thetastar=cosmo_info['thetastar'],
    )
    camb_pars.DarkEnergy = DarkEnergyPPF(w=cosmo_info['w0'], wa=cosmo_info['wa'])
    return camb_pars, cosmo_info

## this function varies the HOD params given the config file
## this function is called in the loop for the param vals
def vary_HOD_param(config, tracer='LRG', param=None, param_val=None):
    if param is None:
        return config['HOD_params']
    HOD_params = copy.deepcopy(config['HOD_params'])
    HOD_params[f'{tracer}_params'][param] = param_val
    return HOD_params

def mock2fits(mock_dict, tracer, filepath, mock_info, cosmo_info, HOD_info):
    col_names = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'mass')
    cols = [mock_dict[tracer][k] for k in col_names]
    t = Table(cols, names=col_names)
    t.meta['MOCK_INFO']  = json.dumps(mock_info)
    t.meta['COSMO_INFO'] = json.dumps(cosmo_info)
    t.meta['HOD_INFO']   = json.dumps(HOD_info)
    t.write(filepath, format='fits', overwrite=True)

## load baseline HOD params from lrg_params.csv for given tracer/model/z combination
## satellite profile params (s, s_v, s_p, s_r, Acent, Asat = 0) and incompleteness (ic = 1)
def load_lrg_hod_params(csv_path, tracer, model, z):
    hod_table = load_hod_table(csv_path)
    key = (tracer, model, float(z))
    if key not in hod_table:
        raise ValueError(f'no HOD params found for tracer={tracer}, model={model}, z={z} in {csv_path}')
    return asdict(hod_table[key])

## build the full AbacusHOD config dict (sim_params + HOD_params) from config.yaml + CSV
def build_abacus_config(sim_name, z_mock, tracer, hod_model):
    lrg_params    = load_lrg_hod_params(HOD_CSV, tracer=tracer, model=hod_model, z=z_mock)
    sim_dir       = PROJECT_ROOT / 'abacus'
    subsample_dir = PROJECT_ROOT / 'subsamples/'
    mock_dir    = PROJECT_ROOT / 'results' / 'mocks'
    return {
        'sim_params': {
            'sim_name':      sim_name,
            'sim_dir':       str(sim_dir),
            'subsample_dir': str(subsample_dir),
            'output_dir':    str(mock_dir),
            'z_mock':        float(z_mock),
        },
        'HOD_params': {
            'tracer_flags': {tracer: True, 'ELG': False, 'QSO': False},
            'want_ranks':   True,    ## required for s_v satellite profile variation
            'want_AB':      False,
            'want_rsd':     True,
            'want_shear':   False,
            f'{tracer}_params': lrg_params,
        },
    }

## writes the HOD config dict to config/hod/, creating parent dirs as needed
def write_hod_config(config, sim_type, tracer):
    config_dir = PROJECT_ROOT / 'config' / 'hod'
    config_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = config_dir / f'{tracer}-config-{sim_type}.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f'HOD config written to {yaml_path}')
    return yaml_path

## prepare_sim generates the HDF5 subsamples if they don't already exist
def ensure_subsamples(sim_params, hod_flags, n_threads):
    subsample_path = (Path(sim_params['subsample_dir'])
                      / sim_params['sim_name']
                      / f"z{sim_params['z_mock']:.3f}")
    if subsample_path.exists() and any(subsample_path.glob('*.h5')):
        print(f'subsamples already exist at {subsample_path}! skipping prepare_sim')
        return

    from abacusnbody.hod.prepare_sim import main as prepare_main

    print(f'no subsamples found at {subsample_path}, running prepare_sim...')
    prepare_cfg = {
        'sim_params':  {**sim_params, 'cleaned_halos': 1},  ## remove unphysical CompaSO fragments
        'HOD_params':  {
            'tracer_flags': hod_flags['tracer_flags'],
            'want_ranks':   hod_flags['want_ranks'],
            'want_AB':      hod_flags['want_AB'],
            'want_shear':   hod_flags['want_shear'],
            'want_rsd':     hod_flags['want_rsd'],
        },
        'prepare_sim': {'Nthread_per_load': n_threads, 'Nparallel_load': 1},
    }
    config_dir = PROJECT_ROOT / 'config' / 'sims'
    config_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = config_dir / 'prepare_sim_auto.yaml'
    with open(cfg_path, 'w') as f:
        yaml.dump(prepare_cfg, f, default_flow_style=False)
    print(f'prepare_sim config written to {cfg_path}')
    prepare_main(str(cfg_path))
    print('prepare_sim done')


# -----------
# MAIN
# -----------
def main():
    parser = argparse.ArgumentParser(description='generate galaxy mock catalogs w/ abacushod')
    parser.add_argument('--test', action='store_true',
                        help='run base-case mock only')
    args = parser.parse_args()

    cfg       = load_config()
    sim_name  = cfg['sim']['name']
    sim_type  = cfg['sim']['type']
    z_mock    = float(cfg['sim']['z'])
    tracer    = cfg['tracer']
    hod_model = cfg['hod_model']
    n_mocks   = cfg['n_mocks']
    seed_base = cfg['seed_base']
    n_threads = cfg.get('n_threads', 32)
    param_dict = get_param_dict(cfg)

    seed_vals = np.asarray([int(seed_base + i) for i in range(n_mocks)])

    if args.test:
        print('='*50)
        print('TEST MODE: running single base case, no HOD variation')
        print(f'  sim_name = {sim_name}')
        print(f'  z_mock   = {z_mock}')
        print('='*50)
        print()
        param_dict = {'base': [None]}
        n_mocks    = 1
        seed_vals  = seed_vals[:1]

    ## build AbacusHOD config from config.yaml + CSV and write it to config/hod/
    abacus_cfg = build_abacus_config(sim_name, z_mock, tracer, hod_model)
    write_hod_config(abacus_cfg, sim_type, tracer)

    sim_params = abacus_cfg['sim_params']
    print('using the following Abacus sim params for the mocks')
    pprint(sim_params)
    print()

    ## run prepare_sim to generate HDF5 subsamples if they do not exist
    ensure_subsamples(sim_params, abacus_cfg['HOD_params'], n_threads)

    abacus_pars = get_abacus_pars(
        f"{sim_params['sim_dir']}/{sim_params['sim_name']}/abacus.par"
    )

    camb_pars, cosmo_info = abacus_pars2camb_pars(abacus_pars)
    print('cosmology params for the mock [found in the header for fits file]')
    pprint(cosmo_info)
    print()

    cosmo = camb.get_results(camb_pars)

    z  = sim_params['z_mock']
    H0 = cosmo.hubble_parameter(0.)
    Hz = cosmo.hubble_parameter(z)

    mock_info = {
        'boxsize': abacus_pars['BoxSize'],
        'z':  z,
        'a':  1. / (1. + z),
        'H0': H0,
        'h0': H0 / 100.,
        'Hz': Hz,
        'hz': Hz / 100.,
    }

    print('general info about the mocks [found in the header for fits file]')
    pprint(mock_info)
    print()

    for param_name, param_vals in param_dict.items():
        param = None if param_name == 'base' else param_name
        for i, val in enumerate(param_vals):
            HOD_params = vary_HOD_param(abacus_cfg, tracer=tracer, param=param, param_val=val)

            print('-'*10 + f' varying {param_name} @ {val} [{i+1} of {len(param_vals)}] ' + '-'*10)
            print('using the following HOD params')
            pprint(HOD_params)
            print()

            ## now we initialize the HOD Ball object for the given set of params
            print('initializing the HOD Ball object')
            init_start = time.time()
            Ball = AbacusHOD(sim_params, HOD_params)
            print('creating the HOD Ball took {}'.format(time.time() - init_start))
            print()

            ## for a given Ball object, run HOD to generate n_mocks mocks with synced seeds
            mocks_start = time.time()

            for sim_idx in range(n_mocks):
                seed = seed_vals[sim_idx]
                print('-'*50)
                print(f'making mock {sim_idx+1} of {n_mocks}')

                mock_dict = Ball.run_hod(Ball.tracers, want_rsd=False, Nthread=n_threads,
                                         reseed=seed, verbose=True)

                ## adding central and satellite galaxy info to the mock_info dictionary
                mock_info['Ncent'] = mock_dict[tracer]['Ncent']
                mock_info['Nsat']  = len(mock_dict[tracer]['x']) - mock_dict[tracer]['Ncent']

                filepath = mock_path(tracer, param, val, seed, sim_type)
                filepath.parent.mkdir(parents=True, exist_ok=True)  ## auto-create results/mocks/ subdirs

                mock2fits(mock_dict, tracer, filepath, mock_info, cosmo_info,
                          HOD_params[f'{tracer}_params'])

                print(f'mock saved @ {filepath}')
                print('-'*50)
            print(f'making {n_mocks} mocks for {param_name}={val} took {time.time()-mocks_start}s')
            print()

    print('script ran successfully!')


if __name__ == '__main__':
    main()
