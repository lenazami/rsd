# rsd
compressing redshift space distortions using AbacusSummit galaxy mock catalogs.

## Data
- using abacussummit simulations hosted on NERSC under the desi collaboration, check the [data access webpage](https://abacussummit.readthedocs.io/en/latest/data-access.html) for further details
- `data/lrg_params.csv` HOD parameters come from DESI DR2 AbacusHF-v2 mock production documentation (CrossAnalysisInfrastructureWG/LSSMocks/DR2HFAbacusMocks).
- mock catalog gen code originally written by Anmol Raina (harvard physics), modified by me

tested everything first using:
(root: `/global/homes/l/lenazami/rsd/`)
- `abacus/AbacusSummit_highbase_c000_ph100/halos/z0.500/`
    - `halo_rv_A`
    - `halo_info`




## Setup
```bash
pip install -e .
```

## Pipeline
abacus requires yamls in order to perform its things, so we store any and all needed config files under `config/`, except of course OUR own config file, which is at the repo root. 

all parameters are set in `config.yaml`. run scripts in order from the repo root:


Analysis outputs are written to `results/` (gitignored). Auto-generated AbacusHOD YAML configs are written to `config/` (gitignored).

### Catalog generation

```bash
python scripts/catalogs/prep_sim.py
python scripts/catalogs/mocks.py
python scripts/catalogs/mocks.py --n-mocks 5
python scripts/catalogs/mocks.py --vary
python scripts/analysis/clustering.py --variation base   # two-point correlation functions
```
The available flags on mocks.py are  which 

| flag | description |
| :--: | :-- |
| `--n-mocks` | determines the number of mock catalogs will be built per HOD |
| `--vary` | generates HOD variations | 

under `hod_variation.py` 

filename suffixes need to match abacusnbody.hod.prepare_sim.prepare_slab 

| flag | description |
| :--: | :-- |
| `want_rsd` | ??? |
| `want_ranks` |  | 
| `want_AB` | want full 10% subsample rather than 3% from just A | 


| suffix | description |
| :--: | :-- |
| `_MT` | ELG/QSO tracers |
| `_withranks` | on particles when want_ranks | 


## Tree

<!-- ```
config.yaml          # edit this to configure the pipeline
src/utils.py         # shared utilities (I/O, coordinate transforms, paths)
scripts/
  catalogs/
    prep_sim.py          # AbacusSummit prepare_sim (HDF5 subsamples)
    mocks.py             # HOD mock catalog generation
  analysis/
    clustering.py        # two-point correlation functions
data/
  lrg_params.csv         # HOD baseline parameters (DESI DR2)
  bins/                  # bin edge files for clustering (??)
  subsamples/            # HDF5 particle subsamples from prepare_sim (gitignored)
  mocks/
``` -->

```
config.yaml # edit this to configure the entire pipeline
README.md # you're here!
config/
├── hod/
└── sims/
src/
└── utils.py      # io, coordinate transforms, pathfinding, etc.
scripts/
├── analysis/
│   ├── clustering.py      # two-point correlation functions
└── catalogs/
    ├── hod_variation.py   # generates hod variations using sobol sampling
    ├── mocks.py           # generates mocks from subsamples
    └── prep_sim.py        # generates subsamples from raw abacus data
```
