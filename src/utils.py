# src/utils.py

# -----------
# IMPORTS
# -----------
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.table import Table
import json
import time
import yaml
from pathlib import Path

from pycorr import TwoPointCorrelationFunction

# -----------
# PATHS
# -----------
## PROJECT_ROOT is the repo root (one level up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HOD_CSV      = PROJECT_ROOT / 'data/lrg_params.csv'
COORDS = ('x', 'y', 'z')

# -----------
# CONFIG
# -----------
def load_config(config_path=None):
    """Load config.yaml from the project root (or a custom path)."""
    if config_path is None:
        config_path = PROJECT_ROOT / 'config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)

def get_param_dict(cfg):
    """Build the HOD variation dict from config: {'base': [None], 'alpha_c': [...], ...}"""
    variations = cfg.get('hod_variations', {})
    return {'base': [None], **variations}

# -----------
# FUNCTIONS
# -----------
def _get_positions(mock, fpar=1., fperp=1., rsd=False):
    suffix = '_rsd' if rsd else ''
    
    positions = np.empty((3, len(mock)), dtype='f8')
    positions[0] = mock[f'x{suffix}']
    positions[1] = mock[f'y{suffix}']
    positions[2] = mock[f'z{suffix}']

    if fpar!=1. or fperp!=1.:
        positions[:2] /= fperp
        positions[2] /= fpar
    
    return positions
    
def _mock_info(table, *keys):
    mock_info = json.loads(table.meta['MOCK_INFO'])
    
    if not keys:
        return mock_info
    if len(keys) == 1:
        return mock_info[keys[0]]
    return tuple(mock_info[key] for key in keys)

def _correlation(mode, mock, edges, rsd, fpar, fperp, nthreads=32):
    positions = _get_positions(mock, rsd=rsd, fpar=fpar, fperp=fperp)
    L = _mock_info(mock, 'boxsize')
    boxsize = [L / fperp, L / fperp, L / fpar]

    return TwoPointCorrelationFunction(mode=mode, edges=edges,
                                       data_positions1=positions,
                                       data_positions2=positions,
                                       engine='corrfunc',
                                       nthreads=nthreads,
                                       los='z',
                                       boxsize=boxsize)
    
## all useful functions for manipulating the data are here
## function to load the data into astropy tables
## mocks live under results/mocks/<param>/<tracer>mock-<tag>-seed=<seed>-<sim_type>.fits
def mock_path(tracer, param, param_value, seed, sim_type):
    base = param is None
    sub  = 'base' if base else param
    tag  = 'base' if base else f'{param}={param_value}'
    return PROJECT_ROOT / 'results' / 'mocks' / sub / f'{tracer}mock-{tag}-seed={seed}-{sim_type}.fits'


def get_mock(tracer, seed=101, param=None, param_value=None, sim_type='fiducial'):
    return Table.read(mock_path(tracer, param, param_value, seed, sim_type))


## function to change positions to match corrfunc
def update_coords(table):
    boxsize = _mock_info(table, 'boxsize')
    for coord in COORDS:
        table[coord] += boxsize/2.

    return table


## function to add an RSD column to the data
def add_rsd(table,los='z'): 
    if los not in COORDS:
        raise ValueError('los must be one of {}'.format(COORDS))

    boxsize, Hz, a = _mock_info(table, 'boxsize', 'Hz', 'a')

    for coord in COORDS:
        rsd = table[coord]
        if coord == los:
            rsd = rsd + table[f'v{coord}']/(Hz*a)
        table[f'{coord}_rsd'] = rsd % boxsize
        
    return table

def load_mock(tracer, seed, param, param_value, sim_type, rsd=True):
    mock = get_mock(tracer=tracer, seed=seed, param=param,
                    param_value=param_value, sim_type=sim_type)
    mock = update_coords(mock)
    if rsd:
        mock = add_rsd(mock, los='z')
    return mock


## function to split table into centrals and satellites
def split_censat(table):
    Ncent, Nsat = _mock_info(table, 'Ncent', 'Nsat')

    print('splitting the mock into {} centrals and {} satellites'.format(Ncent,Nsat))
    
    cents = table[:Ncent]
    sats = table[Ncent:]

    return cents, sats


## function to make a cutout from the mock - assuming corrfunc coords (use after applying "update_coords")
def get_cutout(table,size,center):
    if table['x'].min() < 0.:
        print('need to first use "update_coords"')
        return
    
    size, center = np.asarray(size), np.asarray(center)
    
    boxsize = _mock_info(table, 'boxsize')
    lower = center - size/2.
    upper = center + size/2.
    
    if np.any(lower < 0.) or np.any(upper > boxsize):
        print('cutout is out of bounds')
        return

    mask = (
        (table['x'] >= lower[0]) & (table['x'] <= upper[0]) &
        (table['y'] >= lower[1]) & (table['y'] <= upper[1]) &
        (table['z'] >= lower[2]) & (table['z'] <= upper[2])
    )

    cutout = table[mask]
    return cutout

def paramvar_mocks(tracer='LRG', seed=101, param_dict=None, sim_type='fiducial', rsd=True):
    if param_dict is None:
        param_dict = {'base': [None]}
    data_dict = {}
    for key in param_dict:
        param = None if key == 'base' else key
        data_dict[key] = [load_mock(tracer=tracer, seed=seed, param=param,
                                    param_value=pv, sim_type=sim_type, rsd=rsd)
                          for pv in param_dict[key]]
    return data_dict


def mock2xi(mock, corr_type, edges, pimax=None, ells=None,
            rsd=True, fpar=1., fperp=1., nthreads=32):
    start = time.time()

    mode_kwargs = {
        'rppi': {'pimax': pimax},
        'smu': {'ells': ells},
    }
    
    if corr_type not in mode_kwargs:
        raise ValueError('corr_type not supported :(')
    
    result = _correlation(corr_type, mock, edges, rsd, fpar, fperp, nthreads=nthreads)
    xi_2d_vals = result.get_corr()
    _, xi_1d_vals = result(return_sep=True, **mode_kwargs[corr_type])

    print('took {}s to compute xi'.format(time.time()-start))
    return xi_2d_vals, xi_1d_vals

def ret_APparams(results_true,results_assumed,z):
    
    Hz_t = results_true.hubble_parameter(z) * u.km/u.s/u.Mpc
    Hz_a = results_assumed.hubble_parameter(z) * u.km/u.s/u.Mpc
    
    r_par_t = (const.c/Hz_t).to(u.Mpc)
    r_par_a = (const.c/Hz_a).to(u.Mpc)

    r_perp_t = results_true.comoving_radial_distance(z) * u.Mpc
    r_perp_a = results_assumed.comoving_radial_distance(z) * u.Mpc
    
    fpar = r_par_t/r_par_a
    fperp = r_perp_t/r_perp_a
    
    alpha_iso = (fpar*fperp**2)**(1/3)
    alpha_AP = fpar/fperp
    
    return fpar.value, fperp.value, alpha_iso.value, alpha_AP.value

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

class HODDatabase:

    def __init__(self, csv_path):
        self._params = load_hod_table(csv_path)

    def get(self, tracer, model, z):
        return self._params[(tracer, model, float(z))]
    
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
