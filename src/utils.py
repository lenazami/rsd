# src/utils.py

# -----------
# IMPORTS
# -----------
from dataclasses import dataclass
import numpy as np
from astropy.table import Table
import json
import yaml
from pathlib import Path

# -----------
# PATHS
# -----------
# PROJECT_ROOT is the repo root (one level up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
COORDS = ('x', 'y', 'z')

@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data(self):
        return self.root / 'data'

    @property
    def results(self):
        return self.root / 'results'

    @property
    def config_dir(self):
        return self.root / 'config'

    @property
    def abacus(self):
        return self.root / 'abacus'

    @property
    def mocks(self):
        return self.data / 'mocks'

    @property
    def subsamples(self):
        return self.data / 'subsamples'

    @property
    def hod_csv(self):
        return self.data / 'lrg_params.csv'

    def config(self, config_path=None):
        if config_path is None:
            return self.root / 'config.yaml'
        return Path(config_path)

    def hod_var(self, variation_name):
        return self.data / 'hod_vars' / f'{variation_name}.csv'

    def hod_config(self, sim_type, tracer):
        return self.config_dir / 'hod' / f'{tracer}-config-{sim_type}.yaml'

    def prepare_sim_config(self, sim_name, z_mock):
        return self.config_dir / 'sims' / f'prepare_sim_{sim_name}_z{z_mock:.3f}.yaml'

    def halo_info(self, sim_name, z_mock):
        return self.abacus / sim_name / 'halos' / f'z{z_mock:.3f}' / 'halo_info'

    def subsample_set(self, sim_name, z_mock):
        return self.subsamples / sim_name / f'z{z_mock:.3f}'

    def mock(self, tracer, seed, sim_type, variation_id=None):
        if variation_id is None:
            return self.mocks / 'base' / f'{tracer}mock-base-seed={seed}-{sim_type}.fits'
        return (self.mocks / 'variation' / f'{tracer}mock-variation-{variation_id}'
                / f'seed={seed}-{sim_type}.fits')

    def bins(self, corr_type, regime):
        return self.data / 'bins' / f'{corr_type}_bins_{regime}.npz'

    def xi(self, mode, corr_type, sim_type, regime, rsd, fpar):
        ap = (fpar != 1)
        tags = [tag for tag, active in (('rsd', rsd), ('AP', ap)) if active]
        suffix = '_' + '+'.join(tags) if tags else ''
        filename = f'xi_{corr_type}_{mode}_{sim_type}_{regime}{suffix}.pkl'
        return self.results / 'xi' / f'xi_{mode}' / filename


PATHS = ProjectPaths(PROJECT_ROOT)

# -----------
# FUNCTIONS
# -----------
def load_config(config_path=None):
    """Load config.yaml from the project root (or a custom path)."""
    with open(PATHS.config(config_path)) as f:
        return yaml.safe_load(f)

# the abacushod sim_params block, shared by mocks.py and prep_sim.py
def build_sim_params(sim_name, z_mock):
    return {
        'sim_name':      sim_name,
        'sim_dir':       str(PATHS.abacus),
        'subsample_dir': str(PATHS.subsamples),
        'output_dir':    str(PATHS.mocks),
        'z_mock':        float(z_mock),
    }

# the abacushod HOD flags block (tracer_flags + want_* switches), shared by
def build_hod_flags(tracer):
    return {
        'tracer_flags': {tracer: True, 'ELG': False, 'QSO': False},
        'want_ranks':   True,    # required for s_v satellite profile variation
        'want_AB':      False,
        'want_rsd':     False,
        'want_shear':   False,
    }

def mock_info(table, *keys):
    info = json.loads(table.meta['MOCK_INFO'])
    
    if not keys:
        return info
    if len(keys) == 1:
        return info[keys[0]]
    return tuple(info[key] for key in keys)

# all useful functions for manipulating the data are here
# function to load the data into astropy tables
def get_mock(tracer, seed=101, sim_type='fiducial', variation_id=None):
    return Table.read(PATHS.mock(tracer, seed, sim_type, variation_id))

# function to change positions to match corrfunc
def update_coords(table):
    boxsize = mock_info(table, 'boxsize')
    for coord in COORDS:
        table[coord] += boxsize/2.

    return table

# function to add an RSD column to the data
def add_rsd(table,los='z'): 
    if los not in COORDS:
        raise ValueError('los must be one of {}'.format(COORDS))

    boxsize, Hz, a, h0 = mock_info(table, 'boxsize', 'Hz', 'a', 'h0')

    for coord in COORDS:
        rsd = table[coord]
        if coord == los:
            rsd = rsd + table[f'v{coord}']*h0/(Hz*a)
        table[f'{coord}_rsd'] = rsd % boxsize
        
    return table

def load_mock(tracer, seed, sim_type, variation_id=None, rsd=True, los='z'):
    mock = get_mock(tracer=tracer, seed=seed, sim_type=sim_type, variation_id=variation_id)
    mock = update_coords(mock)
    if rsd:
        mock = add_rsd(mock, los=los)
    return mock
