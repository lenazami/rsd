# src/utils.py

# -----------
# IMPORTS
# -----------
from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np
import pandas as pd
from astropy.table import Table
import json
import yaml

# -----------
# PATHING
# -----------
# constants
TRACERS = {"LRG", "ELG", "QSO"}
SIM_DEFAULTS = {
    "highbase": {"cosmo": "000", "phase": "100"},
    "base": {"cosmo": "000", "phase": "000"},
}

# helper fns

def tracer_name(tracer: str) -> str:
    tracer = tracer.upper()
    if tracer not in TRACERS:
        raise ValueError(f"Unknown tracer {tracer!r}; expected one of {sorted(TRACERS)}")
    return tracer

def active_suffix(**flags: bool) -> str:
    tags = [name for name, active in flags.items() if active]
    return "_" + "+".join(tags) if tags else ""

def ztag(z) -> str:
    return f"z{float(z):.3f}"

# dataclasses
@dataclass(frozen=True)
class ResultsPaths:
    root: Path

    @property
    def figs(self):
        return self.root / "figs"

    @property
    def xi(self):
        return self.root / "xi"

    @property
    def pwv(self):
        return self.root / "pwv"

    # xi pickle, grouped per hod under results/xi/xi_<hod>/
    def xi_file(self, corr_type, regime, hod, rsd=False, fpar=1.) -> Path:
        suffix = active_suffix(rsd=rsd, ap=(fpar != 1.))
        return self.xi / f"xi_{hod}" / f"xi_{corr_type}_{hod}_{regime}{suffix}.pkl"

    # 1d 2pcf figure under results/figs/2pcf/ (same stem as the xi pickle)
    def fig_2pcf(self, corr_type, regime, hod, rsd=False, fpar=1.) -> Path:
        suffix = active_suffix(rsd=rsd, ap=(fpar != 1.))
        return self.figs / "2pcf" / f"xi_{corr_type}_{hod}_{regime}{suffix}.png"

@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def data(self):
        return self.root / "data"
    
    @property
    def abacus(self):
        return self.root / "abacus"
    
    @property
    def mocks(self):
        return self.data / "mocks"

    @property
    def subsamples(self):
        return self.data / "subsamples"
    
    @property
    def hods(self):
        return self.data / "hods"

    def subsample_dir(self, sim: "SimID") -> Path:
        return self.subsamples / sim.realization / sim.ztag

    def halo_info(self, sim: "SimID") -> Path:
        return self.abacus / sim.realization / "halos" / sim.ztag / "halo_info"

    # baseline (fiducial) HOD params table, one row per (model, z); keyed by tracer only
    def hod_csv(self, mock: "MockID") -> Path:
        return self.hods / "fids" / f"{mock.tracer}_params.csv"

    # Sobol-sampled HOD variation csv
    def hod_var(self, name: str) -> Path:
        return self.hods / "vars" / f"{name}.csv"

    # mock catalog, grouped per hod model under data/mocks/<hod_model>/
    def mock_fits(self, mock: "MockID") -> Path:
        fname = f"mock_{mock.tracer}_{mock.hod_family}_h{mock.hod_id}_seed{mock.seed}.fits"
        return self.mocks / mock.hod_model / fname

@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def from_file(cls, file: str | Path) -> "ProjectPaths":
        return cls(Path(file).resolve().parent.parent)

    @property
    def data(self):
        return DataPaths(self.root)

    @property
    def results(self):
        return ResultsPaths(self.root / "results")

    @property
    def cfg(self) -> Path:
        return self.root / 'config.yaml'

PATHS = ProjectPaths.from_file(__file__)

@dataclass(frozen=True)
class SimID:
    name : str          # "base", "highbase"
    z    : float
    cosmo: str = "000"
    phase: str = "000"

    @classmethod
    def cfg(cls, config, **overrides):
        
        if config is None:
            config=load_config()
        sim = dict(config.get("sim_id", {}))
        sim.update({k: v for k, v in overrides.items() if v is not None})
        name = sim.get("name", sim.get("name"))

        if name not in SIM_DEFAULTS:
            raise ValueError(f"Unknown sim name: {name}")

        for key, value in SIM_DEFAULTS[name].items():
            sim.setdefault(key, value)

        return cls(
            name=name,
            z=float(sim["z"]),
            cosmo=str(sim["cosmo"]).zfill(3),
            phase=str(sim["phase"]).zfill(3),
        )

    @property
    def suite(self):
        return f"AbacusSummit_{self.name}"

    @property
    def realization(self):
        return f"{self.suite}_c{self.cosmo}_ph{self.phase}"

    @property
    def ztag(self):
        return ztag(self.z)
    
    def sim_params(self):
        return {
            "sim_name": self.realization,
            "sim_dir": str(PATHS.data.abacus),
            "subsample_dir": str(PATHS.data.subsamples),
            "output_dir": str(PATHS.data.mocks),
            "z_mock": self.z,
        }
    
@dataclass(frozen=True)
class MockID:
    tracer: str
    hod_family: str = "fid"
    hod_id: str = "000"
    seed: int = 100
    hod_model: str = "base"

    def __post_init__(self):
        object.__setattr__(self, "tracer", tracer_name(self.tracer))

    @classmethod
    def from_config(cls, config):
        mock = config["mock"]
        return cls(
            tracer=mock["tracer"],
            hod_family=mock.get("hod_family", "fid"),
            hod_model=config.get("hod_model", "base"),
        )
        
    @property
    def param_key(self):
        return f"{self.tracer}_params"
    
    def hod_flags(self):
        return {
            "tracer_flags": {
                "LRG": self.tracer == "LRG",
                "ELG": self.tracer == "ELG",
                "QSO": self.tracer == "QSO",
            },
            "want_ranks": True, # required for s_v satellite profile variation
            "want_AB": False,
            "want_rsd": False,
            "want_shear": False,
        }

    def hod_params(self, z):
        df = pd.read_csv(PATHS.data.hod_csv(self))
        mask = (
            (df["TRACER"].map(tracer_name) == self.tracer)
            & (df["ZSNAP"].astype(float).map(ztag) == ztag(z))
            & (df["MODEL"] == self.hod_model)
        )
        rows = df.loc[mask]
        pars = rows.iloc[0].drop(labels=["TRACER", "MODEL", "ZSNAP"]).astype(float).to_dict()
        return {
                **self.hod_flags(),
                self.param_key: pars,
            }

    # function to load the data into astropy tables
    def load(self, rsd=False, los='z'):
        tab = Table.read(PATHS.data.mock_fits(self))
        tab = update_coords(tab)
        if rsd:
            tab = add_rsd(tab, los=los)
        return tab

# -----------
# DATA LOADING+MANIPULATION
# -----------
COORDS = ('x', 'y', 'z')

def load_config():
    """Load config.yaml from the project root"""
    with open(PATHS.cfg) as f:
        return yaml.safe_load(f)

# to print out data in a neater format
def summarize(data):
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: ndarray shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, tuple):
            print(f"{k}: tuple")
            for i, x in enumerate(v):
                if isinstance(x, np.ndarray):
                    print(f"    [{i}] shape={x.shape}, dtype={x.dtype}")
                else:
                    print(f"    [{i}] {type(x).__name__}")
        else:
            print(f"{k}: {v}")
            
def mock_info(table, *keys):
    info = json.loads(table.meta['MOCK_INFO'])
    
    if not keys:
        return info
    if len(keys) == 1:
        return info[keys[0]]
    return tuple(info[key] for key in keys)

# all useful functions for manipulating the data are here
# function to change positions to match corrfunc
def update_coords(table):
    boxsize = mock_info(table, 'boxsize')
    for coord in COORDS:
        table[coord] += boxsize/2.
    
    return table

# function to add an RSD column to the data
def add_rsd(table,los='z'): 
    if los not in COORDS:
        raise ValueError(f'los must be one of {COORDS}')
    
    boxsize, Hz, a, h0 = mock_info(table, 'boxsize', 'Hz', 'a', 'h0')
    
    for coord in COORDS:
        rsd = table[coord]
        if coord == los:
            rsd = rsd + table[f'v{coord}']*h0/(Hz*a)
        table[f'{coord}_rsd'] = rsd % boxsize
        
    return table
