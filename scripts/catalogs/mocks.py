# scripts/catalogs/mocks.py
# gens galaxy mock catalogs w/ Abacushod.
# allows for hod parameter variations defined in config.yaml. 
# TODO: is the config.yaml thing still true?
# default run makes one base-hod mock; use flags for more seeds or variations.

# -----------
# IMPORTS
# -----------
import argparse
import ast
import json
import time
from pprint import pprint

import numpy as np
import yaml
from astropy.table import Table

import camb
from camb.dark_energy import DarkEnergyPPF
from abacusnbody.hod.abacus_hod import AbacusHOD

from src.utils import PATHS, COORDS, load_config, build_sim_params, build_hod_flags

from scripts.catalogs.hod_variation import (hod_var_path, load_hod_var,
                                            load_lrg_hod_params, set_HOD_params)
from scripts.catalogs.prep_sim import check_subsamples

# -----------
# FUNCTIONS
# -----------
# this function parses thru the abacus.pars file
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

# this function scrapes the abacus_pars to get CAMB cosmology params
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
    camb_pars = camb.set_params(**{k: cosmo_info[k] for k in
        ('ombh2', 'omch2', 'omk', 'tau', 'As', 'ns', 'thetastar')})
    camb_pars.DarkEnergy = DarkEnergyPPF(w=cosmo_info['w0'], wa=cosmo_info['wa'])
    return camb_pars, cosmo_info

def mock2fits(mock_dict, tracer, filepath, mock_info, cosmo_info, HOD_info, halo_data):
    gal   = mock_dict[tracer]
    Ncent = gal['Ncent']
    Ntot  = len(gal['x'])

    col_names = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'mass')
    cols = [gal[k] for k in col_names]
    t = Table(cols, names=col_names)

    # host halo id (from run_hod) + centrals: centrals=[:Ncent]
    t['halo_id'] = np.asarray(gal['id'])
    is_central = np.zeros(Ntot, dtype='i4')
    is_central[:Ncent] = 1
    t['is_central'] = is_central

    # join halo centers on host halo id.
    halo_pos, halo_vel = match_halo_centers(halo_data, gal['id'])
    for i, c in enumerate(COORDS):
        t[f'halo_{c}']  = halo_pos[:, i]
        t[f'halo_v{c}'] = halo_vel[:, i]

    t.meta['MOCK_INFO']  = json.dumps(mock_info)
    t.meta['COSMO_INFO'] = json.dumps(cosmo_info)
    t.meta['HOD_INFO']   = json.dumps(HOD_info)
    t.write(filepath, format='fits', overwrite=True)

# run_hod does not output host-halo centers, so we do it ourselves
def match_halo_centers(halo_data, ids):
    hid   = halo_data['hid']
    ids   = np.asarray(ids).astype(hid.dtype, copy=False)
    pos   = np.searchsorted(hid, ids)
    found = (pos < len(hid)) & (hid[np.clip(pos, 0, len(hid) - 1)] == ids)
    if not np.all(found):
        raise ValueError(f'{np.sum(~found)} galaxy host-halo ids not found in halo_data')
    return halo_data['hpos'][pos], halo_data['hvel'][pos]

# build abacushod config dict (sim_params + hod_params) from config.yaml + csv
def build_abacus_config(sim_name, z_mock, tracer, hod_model):
    lrg_params = load_lrg_hod_params(PATHS.hod_csv, tracer=tracer, model=hod_model, z=z_mock)
    return {
        'sim_params': build_sim_params(sim_name, z_mock),
        'HOD_params': {
            **build_hod_flags(tracer, want_rsd=False),
            f'{tracer}_params': lrg_params,
        },
    }

# writes the hod config dict to config/hod/, creating parent dirs as needed
def write_hod_config(config, sim_type, tracer):
    yaml_path = PATHS.hod_config(sim_type, tracer)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f'HOD config written to {yaml_path}')
    return yaml_path

# load cosmology (via CAMB) + general mock info for the fits header
def setup_cosmology(sim_params):
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

    return cosmo_info, mock_info

# -----------
# MAIN
# -----------
def main():
    parser = argparse.ArgumentParser(description='generate galaxy mock catalogs w/ abacushod')
    parser.add_argument('--n-mocks', type=int, default=1,
                        help='number of sequential seeds to run; default is one base-HOD mock')
    parser.add_argument('--vary', action='store_true',
                        help='loop over the sampled HOD variations from config.yaml')
    args = parser.parse_args()

    cfg       = load_config()
    sim_name  = cfg['sim']['name']
    sim_type  = cfg['sim']['type']
    z_mock    = float(cfg['sim']['z'])
    tracer    = cfg['tracer']
    hod_model = cfg['hod_model']
    n_mocks   = args.n_mocks
    seed_base = cfg['seed_base']
    n_threads = cfg.get('n_threads', 32)

    if n_mocks < 1:
        raise ValueError('--n-mocks must be >= 1')

    seed_vals = np.asarray([int(seed_base + i) for i in range(n_mocks)])

    # vars in format (variation_id, overrides)
    if args.vary:
        var_csv = hod_var_path(cfg['hod_variation']['name'])
        if not var_csv.exists():
            raise FileNotFoundError(f'no HOD variation csv at {var_csv}; generate using '
                                    f'scripts/catalogs/hod_variation.py')
        variations = load_hod_var(var_csv)
        print(f'running {len(variations)} variations from {var_csv}')
    else:
        variations = [(None, {})]
        print(f'running base case for {len(seed_vals)} mock(s)')
    print()

    # build Abacushod config from config.yaml + CSV and write it to config/hod/
    abacus_cfg = build_abacus_config(sim_name, z_mock, tracer, hod_model)
    write_hod_config(abacus_cfg, sim_type, tracer)

    sim_params = abacus_cfg['sim_params']
    print('using the following Abacus sim params for the mocks')
    pprint(sim_params)
    print()

    # checks if submsaples don't exist and errors out
    # TODO: might be better to just record the file error and only process existing files?
    complete, missing, n_slabs = check_subsamples(sim_params, abacus_cfg['HOD_params'])
    if not complete:
        raise FileNotFoundError(
            f'subsamples incomplete for {sim_name} z{z_mock:.3f}: {len(missing)} of '
            f'{n_slabs} slabs missing {missing}. run scripts/catalogs/prep_sim.py first.'
        )

    cosmo_info, mock_info = setup_cosmology(sim_params)

    for vi, (variation_id, overrides) in enumerate(variations):
        HOD_params = set_HOD_params(abacus_cfg, tracer=tracer, overrides=overrides)
        label = 'base' if variation_id is None else f'variation {variation_id}'

        print('-'*10 + f' {label} [{vi+1} of {len(variations)}] ' + '-'*10)
        print('using the following HOD params')
        pprint(HOD_params[f'{tracer}_params'])
        print()

        # now we initialize the HOD Ball object for the given set of params
        print('initializing the HOD Ball object')
        init_start = time.time()
        Ball = AbacusHOD(sim_params, HOD_params)
        print('creating the HOD Ball took {}'.format(time.time() - init_start))
        print()

        # for a given Ball object, run HOD to generate a mock per seed with synced seeds
        mocks_start = time.time()

        for sim_idx, seed in enumerate(seed_vals):
            print('-'*50)
            print(f'making mock {sim_idx+1} of {len(seed_vals)} (seed={seed})')

            mock_dict = Ball.run_hod(Ball.tracers, want_rsd=False, Nthread=n_threads,
                                     reseed=seed, verbose=True)

            # adding central and satellite galaxy info to the mock_info dictionary
            mock_info['Ncent'] = mock_dict[tracer]['Ncent']
            mock_info['Nsat']  = len(mock_dict[tracer]['x']) - mock_dict[tracer]['Ncent']

            filepath = PATHS.mock(tracer, seed, sim_type, variation_id=variation_id)
            filepath.parent.mkdir(parents=True, exist_ok=True)  # auto-create data/mocks/ subdirs

            mock2fits(mock_dict, tracer, filepath, mock_info, cosmo_info,
                      HOD_params[f'{tracer}_params'], Ball.halo_data)

            print(f'mock saved @ {filepath}')
            print('-'*50)
        print(f'making {len(seed_vals)} mocks for {label} took {time.time()-mocks_start}s')
        print()

    print('script ran successfully!')

if __name__ == '__main__':
    main()
