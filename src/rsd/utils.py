# src/utils.py

# -----------
# IMPORTS
# -----------
import os
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv
import h5py
import numpy as np
import pandas as pd
import json
import yaml

# -----------
# PATHING
# -----------
# constants
COORDS = ('x', 'y', 'z')
TRACERS = {"LRG", "ELG", "QSO"}
# TODO: lets see if we can kill these constants where possible
SIM_DEFAULTS = {
    "highbase": {"cosmo": "000", "phase": "100"},
    "base": {"cosmo": "000", "phase": "000"},
}

# helper fns
def hex_id(length: int=2) -> str:
    return f'%0{length}x' % random.randrange(16**length)

def tracer_name(tracer: str) -> str:
    tracer = tracer.upper()
    if tracer not in TRACERS:
        raise ValueError(f"bad tracer {tracer!r}; use {sorted(TRACERS)}")
    return tracer

def active_suffix(**flags: bool) -> str:
    tags = [name for name, active in flags.items() if active]
    return "_" + "+".join(tags) if tags else ""

def ztag(z) -> str:
    return f"z{float(z):.3f}"


@dataclass(frozen=True)
class Paths:
    root: Path

    @classmethod
    def from_file(cls, file: str | Path) -> "Paths":
        return cls(Path(file).resolve().parents[2])

    @property
    def scratch(self) -> Path:
        return Path(os.environ["SCRATCH"])

    # --- top-level ---
    @property
    def data(self):
        return self.root / "data"

    @property
    def results(self):
        return self.root / "results"

    @property
    def cfg(self) -> Path:
        return self.config_path()

    def config_path(self, name="config") -> Path:
        return self.root / "config" / f"{name}.yaml"

    # --- data subtree ---
    @property
    def mocks(self):
        return self.data / "mocks"

    @property
    def hods(self):
        return self.data / "hods"

    @property
    def graphs(self):
        return self.data / "graphs"
    
    @property
    def figs(self):
        return self.results / "figs"

    def subsamples(self, scratch=False):
        if scratch:
            return self.scratch / "subsamples"
        return self.data / "subsamples"

    def subsample_dir(self, sim: "SimID", scratch=False) -> Path:
        return self.subsamples(scratch) / sim.realization / sim.ztag

    # abacus root, or a per-sim subpath: "par" → abacus.par, "halo_info" → halo_info dir
    def abacus(self, sim: "SimID" = None, sub: str = None) -> Path:
        base = self.root / "abacus"
        if sub == "halo_info":
            return base / sim.realization / "halos" / sim.ztag / "halo_info"
        if sub == "par":
            return base / sim.realization / "abacus.par"
        return base

    # unified hod params table (ids 000-099 fid, 100-999 var) + its id manifest
    @property
    def hod_csv(self) -> Path:
        return self.hods / "hods.csv"

    @property
    def hod_manifest(self) -> Path:
        return self.hods / "manifest.csv"

    # simulation registry: short hex code -> (name, cosmo, phase) map of record
    @property
    def sim_registry(self) -> Path:
        return self.data / "sims.csv"

    def mock_h5(self, mock: "MockID") -> Path:
        fname = (f"mock_{mock.tracer}_{mock.hod_family}_h{mock.hod_id}"
                 f"_{mock.sim.tag}_seed{mock.seed}.h5")
        return self.mocks / mock.hod_model / fname

    # per-central graph dataset keyed by sim + mock + cylinder half-extents, e.g. cyl2x10
    def graph_npz(self, mock: "MockID", s_max, par_factor) -> Path:
        assert s_max > 0 and par_factor >= 1, \
            f"bad graph key s_max={s_max} par_factor={par_factor}"
        fname = (f"graph_{mock.tracer}_{mock.hod_family}_h{mock.hod_id}_{mock.sim.tag}"
                 f"_seed{mock.seed}_cyl{s_max:g}x{s_max * par_factor:g}.npz")
        return self.graphs / mock.hod_model / fname

    # --- results ---
    @property
    def pwv(self):
        return self.results / "pwv"

    def xi_pkl(self, corr_type, regime, hod, rsd=False, fpar=1.) -> Path:
        suffix = active_suffix(rsd=rsd, ap=(fpar != 1.))
        return self.results / "xi" / f"xi_{hod}" / f"xi_{corr_type}_{hod}_{regime}{suffix}.pkl"

    # timestamped figure path under results/figs/<name>/; extra args are joined
    def fig_save(self, ext: str = "png", **labels) -> Path:
        name = sys._getframe(1).f_code.co_name.removeprefix("plot_")
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        tag = "".join(f"_{v}" for v in labels.values())
        return self.figs / name / f"fig_{name}{tag}_{stamp}.{ext}"

    # most recent saved figure for a given name (for display)
    def fig_latest(self, name, ext="png"):
        return max((self.figs / name).glob(f"fig_{name}*.{ext}"),
                default=None, key=lambda p: p.stat().st_mtime)

PATHS = Paths.from_file(__file__)


@dataclass(frozen=True)
class SimID:
    name : str = "base"  # "base", "highbase"
    z    : float = 0.5
    cosmo: str = "000"
    phase: str = "000"

    @classmethod
    def from_cfg(cls, config, **overrides):
        
        if config is None:
            config=load_config()
        sim = dict(config.get("sim_id", {}))
        sim.update({k: v for k, v in overrides.items() if v is not None})
        name = sim.get("name", sim.get("name"))

        if name not in SIM_DEFAULTS:
            raise ValueError(f"unknown sim name: {name}")

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

    # short registry code: 2-hex cosmo + 3-hex phase (name resolved via data/sims.csv)
    @property
    def code(self):
        assert int(self.cosmo) < 256 and int(self.phase) < 4096
        return f"s{int(self.cosmo):02x}{int(self.phase):03x}"

    # filename key: code + compact redshift, e.g. s00000_z05 for base c000 ph000 z0.5
    @property
    def tag(self):
        zi = round(self.z * 10)
        assert abs(zi - self.z * 10) < 1e-9 and zi < 100, f"z={self.z} is not a x.x snapshot"
        return f"{self.code}_z{zi:02d}"

    def sim_params(self, scratch=False):
        # abacusnbody appends simname/ztag itself, so subsample_dir is the ROOT
        return {
            "sim_name": self.realization,
            "sim_dir": str(PATHS.abacus()),
            "subsample_dir": str(PATHS.subsamples(scratch)),
            "output_dir": str(PATHS.mocks),
            "z_mock": self.z,
        }
    
@dataclass(frozen=True)
class MockID:
    tracer: str = "LRG"
    hod_id: str = "000"
    seed: int = 100
    hod_model: str = "base"
    sim: SimID = SimID()

    # TODO: still dk if theres a better way to make a default HOD
    HOD_DEFAULTS = {
        "s": 0.0, "s_v": 0.0, "s_p": 0.0, "s_r": 0.0,
        "Acent": 0.0, "Asat": 0.0, "Bcent": 0.0, "Bsat": 0.0,
        "ic": 1.0,
    }

    def __post_init__(self):
        object.__setattr__(self, "tracer", tracer_name(self.tracer))
        if not 0 <= int(self.hod_id) <= 999:
            raise ValueError(f"hod_id must be in 000-999, not {self.hod_id!r}")

    # fid ids are reserved to 000-099, variations live in 100-999
    @property
    def hod_family(self):
        return "fid" if int(self.hod_id) < 100 else "var"

    @classmethod
    def from_cfg(cls, config):
        mock = config["mock"]
        return cls(
            tracer=mock["tracer"],
            hod_id=str(mock.get("hod_id", "000")),
            hod_model=mock.get("hod_model", "base"),
            seed=mock.get("seed_base", cls.seed),
            sim=SimID.from_cfg(config),
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
            # TODO: is it better to derive this ^ based on if nonzero s params?
            "want_AB": False,
            "want_rsd": False,
            "want_shear": False,
        }

    ID_COLS = ["id", "tracer", "zsnap"]   # module-level, or inline

    def hod_params(self, z):
        df = pd.read_csv(PATHS.hod_csv, dtype={"id": str})
        mask = (
            (df["id"].astype(int) == int(self.hod_id))
            & (df["zsnap"].astype(float).map(ztag) == ztag(z))
        )
        rows = df.loc[mask]
        assert not rows.empty, f"no hod row for id={self.hod_id} z={z} in {PATHS.hod_csv}"
        row = rows.iloc[0]
        assert tracer_name(row["tracer"]) == self.tracer, \
            f"hod id {self.hod_id} belongs to {row['tracer']}, not {self.tracer}"
        pars = {k: float(v) for k, v in row.drop(labels=self.ID_COLS).items() if pd.notna(v)}
        pars = {**self.HOD_DEFAULTS, **pars}
        return {**self.hod_flags(), self.param_key: pars}

    # mocks are hdf5: (N,3) pos/vel/halo_pos/halo_vel + 1d columns, json attrs
    def load(self, keys: list[str] | None = None):
        with h5py.File(PATHS.mock_h5(self), 'r') as f:
            d = {k: f[k][:] for k in (keys if keys is not None else list(f))}
            d['info'] = json.loads(f.attrs['MOCK_INFO'])
        return d

    # pos shifted+wrapped to [0, L); rsd adds pos_rsd with the los axis displaced
    def load_rsd(self, cols: list[str] = [], rsd=False, los='z'):
        d = self.load(list(dict.fromkeys(['pos', 'vel', *cols])))
        L = d['info']['boxsize']
        for k in ('pos', 'halo_pos'):
            if k in d:
                d[k] = (d[k] + L / 2.) % L
        if rsd:
            ax = COORDS.index(los)
            d['pos_rsd'] = d['pos'].copy()
            d['pos_rsd'][:, ax] = (d['pos'][:, ax]
                                   + d['vel'][:, ax] * d['info']['rsd_factor']) % L
        return d


# the hex code drops the suite name, so the registry is the code -> sim map of record;
# appends on first sight, asserts on conflicting reuse of a code
def register_sim(sim: SimID):
    path = PATHS.sim_registry
    row = {"code": sim.code, "name": sim.name, "cosmo": sim.cosmo, "phase": sim.phase}
    df = pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame(columns=list(row))
    hit = df[df["code"] == sim.code]
    if not hit.empty:
        assert hit.iloc[0].to_dict() == row, f"registry conflict for {sim.code}: {hit.iloc[0].to_dict()} vs {row}"
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([df, pd.DataFrame([row])]).to_csv(path, index=False)


# ----------------------
# DATA LOADING+MANIPULATION
# ----------------------
def load_config(name="config"):
    with open(PATHS.config_path(name)) as f:
        return yaml.safe_load(f)

# to print out data in a neater format
# TODO do i need this?
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

def mock_info(data, *keys):
    # dict from MockID.load, or a legacy astropy table
    info = data['info'] if isinstance(data, dict) else json.loads(data.meta['MOCK_INFO'])

    if not keys:
        return info
    if len(keys) == 1:
        return info[keys[0]]
    return tuple(info[key] for key in keys)

# minimum-image displacement on a periodic box; any shape, e.g. (N, 3) or (N,)
def min_img(d, L):
    return d - L * np.round(d / L)
