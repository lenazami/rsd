# rsd
compressing redshift space distortions using AbacusSummit galaxy mock catalogs.


# Notes to Self

halo catalogue parsing:
https://abacusutils.readthedocs.io/en/latest/compaso.html
https://abacussummit.readthedocs.io/en/latest/data-products.html
https://abacusutils.readthedocs.io/en/latest/tutorials/compaso.html


# Data info
- using abacussummit simulations hosted on NERSC under the desi collaboration, check the [data access webpage](https://abacussummit.readthedocs.io/en/latest/data-access.html) for further details
- `data/lrg_params.csv` HOD parameters come from DESI DR2 AbacusHF-v2 mock production documentation (CrossAnalysisInfrastructureWG/LSSMocks/DR2HFAbacusMocks).
- mock catalog gen code originally written by Anmol Raina (harvard physics), modified by me (important! i use h5 for the mock data rather than fits)

tested everything first using:
(root: `/global/homes/l/lenazami/rsd/`)
- `abacus/AbacusSummit_highbase_c000_ph100/halos/z0.500/`
    - `halo_rv_A`
    - `halo_info`

we only require those two directories, so as long as the sim variation has halo rv info then this model works on it. EVERY simulation has the redshifts `z = 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0, 2.5, 3.0` available for `halo_rv_A` (the halo cat paricle subsample). these values are fixed as to avoid corruption in the scripts--but you are free to change the range within config/data.yaml. We target LRG survey ranges, so we only used redshift range `z=[0.4, 1.4]` 

## Setup

use the script 
```bash
bash scripts/start.sh
```
in order to set up the repo. if you are running this on nersc, then it will auto-set the abacus symlink by autofinding the abacus data under tyhe desi collab. if you are not running on nersc, then worry not! you should be able to query the data from the abacus api (though the data is [heavy](https://abacussummit.readthedocs.io/en/latest/disk-space.html) )

```bash
pip install -e .
```
there seems to be an issue w pycorr/Corrfunc (still determining the source), but requires you install Corrfunc manually (also trying to figure out if this can be made default when installing this repo)

```bash
USE_GPU=0 pip install git+https://github.com/cosmodesi/Corrfunc@desi
```

## Pipeline
abacus requires yamls in order to function, so i configured the scripts to write temporary yamls (all of the information is recoverable without saving the yaml). the only yaml we keep is our global one which lets you control for 
 its things, so we store any and all needed config files under `config/`, except of course OUR own config file, which is at the repo root. 

all parameters are set in `config.yaml`. run scripts in order from the repo root:

Analysis outputs are written to `results/` (gitignored). Auto-generated AbacusHOD YAML configs are written to `config/` (gitignored).

### Catalog generation

```bash
python -m catalogs.mocks
python -m catalogs.mocks --n-mocks 5
python -m catalogs.mocks -n 5   # two-point correlation functions
```
The available flags on mocks.py are  which 

| flag | description |
| :--: | :-- |
| `--n-mocks`, `n` | determines the number of mock catalogs will be built per HOD |
| `--tracer`, `t` | explicitly choose a tracer | 

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

```
README.md # you're here!
├── config.yaml # edit this to configure the entire pipeline
├── data.yaml
├── model.yaml
└── sims/
src/rsd/
├── __init__.py # init file of course
├── graphs.py # functions to take mocks and turn them into graphs
├── models.py # stores the gnn classes
├── plot.py # plotting helpers
├── preprocess.py # taking raw abacus sim data and turning them into mocks
├── stats.py # 2pcf, etc.
├── training.py # training loop
└── utils.py # io, coordinate transforms, pathfinding, etc.
scripts/
├── analysis/
│   ├── clustering.py
├── gen/
│   ├── cats.py # generates the mock catalogs from subsamples
│   ├── hod_variation.py # handles hod variation mocks
│   ├── prepsims_scratch.py # sends all sim subsamples to scratch
│   └── training_data.py # writes the graph data
└── training/
    ├── __pycache__
    │   └── train_gnn.cpython-312.pyc
    └── train_gnn.py
```

## Notation

| var_name | coordinate | description |
| :--: | :--: | :-- |
| `(s,mu)` | $s$ | the pair separation |
| | $\mu$ | $\cos\theta=s_\parallel/s$ |
| `(tr,ls)` | $r_t$ | the projected transverse separation; also known as $r_\perp$, $r_{\rm transverse}$ |
| | $r_l$ | the line of sight separation $r_{\rm LOS}$, commonly referred to as $\pi$ in the literature |

We generally use these notations to refer to real space and redshift space:

| var_name | when |
| :--: | :--: |
| `(real,rsd)` | code, or var names. sometimes abbv `(rl,rsd)` |
| $\mathbb{R}$, $\mathbb{S}$ | mathematical space |
| $r_i,s_i$ | r- and s-space LOS positions for the $i^{\rm th}$ galaxy, $g_i$ in units of ($\Delta x, \Delta y, \Delta z$) defined wrt central galaxy |
| real-space, redshift-space | prose |

 as r- and s-space respectively. We also refer to the simulation box directions as $(\hat{x},\hat{y},\hat{z})$ to avoid confusion with $z$ the redshift.


---

# customizations

adding features:



*all figs require `plot_` in front of them in order to properly parse the name, since plots are named after their parent function name


# ml

run names:
- `ml_{model}_{target}_{run_id}`

`{stat}_{tracer}_{cosmo}`

# disk space

for `AbacusSummit_base_c000_ph000` full particles:

``` bash
512	AbacusSummit_base_c000_ph000/abacus.par
512	AbacusSummit_base_c000_ph000/checksums.crc32
3.0T	AbacusSummit_base_c000_ph000/halos
18K	AbacusSummit_base_c000_ph000/info
1.2T	AbacusSummit_base_c000_ph000/lightcones
5.3G	AbacusSummit_base_c000_ph000/log
160K	AbacusSummit_base_c000_ph000/status.log
```

but for our use:
``` bash
447M	graphs/base/old_format
513M	graphs/base
13K	    hods/fids
14K	    hods
216M	mocks/base
2.4G	subsamples/AbacusSummit_highbase_c000_ph100/z0.500
```