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

```bash
python scripts/01_prepare_sim.py       # build HDF5 particle subsamples (skips if done)
python scripts/02_generate_mocks.py    # generate HOD mock catalogs
```

Outputs are written to `results/` (gitignored). Auto-generated AbacusHOD YAML configs are written to `config/` (gitignored).

## Structure
```
config.yaml          # edit this to configure the pipeline
src/utils.py         # shared utilities (I/O, coordinate transforms, clustering, AP params)
scripts/
  01_prepare_sim.py      # AbacusSummit prepare_sim (HDF5 subsamples)
  02_generate_mocks.py   # HOD mock catalog generation
  03_compute_clustering.py  # two-point correlation functions
data/
  lrg_params.csv         # HOD baseline parameters (DESI DR2)
  bins/                  # bin edge files for clustering (??)
```
