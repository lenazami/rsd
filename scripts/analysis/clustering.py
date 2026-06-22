# scripts/analysis/clustering.py
# two-point corr functions from generated mock catalogs
# pass --variation to select HOD variation

# -----------
# IMPORTS
# -----------
import argparse
import pickle
import time

import numpy as np

from src.utils import PATHS, load_config, load_mock, mock_info
from scripts.catalogs.hod_variation import hod_var_path, load_hod_var
from pycorr import TwoPointCorrelationFunction

# -----------
# FUNCTIONS
# -----------
# helpers
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


def _output_meta(sim_type, mode, corr_type, regime, edges, centers, rsd, fpar):
    ap = (fpar != 1)
    return {
        'sim_type':      sim_type,
        'var_HOD_param': mode,
        'corr_type':     corr_type,
        'bin_type':      regime,
        'edges':         edges,
        'centers':       centers,
        'rsd':           rsd,
        'AP':            ap,
    }


def _load_edges(corr_type, regime):
    bin_data = np.load(PATHS.bins(corr_type, regime))
    edges   = (bin_data['edges0'],   bin_data['edges1'])
    centers = (bin_data['centers0'], bin_data['centers1'])
    return edges, centers

def _correlation(mode, mock, edges, rsd, fpar, fperp, nthreads=32):
    positions = _get_positions(mock, rsd=rsd, fpar=fpar, fperp=fperp)
    L = mock_info(mock, 'boxsize')
    boxsize = [L / fperp, L / fperp, L / fpar]

    return TwoPointCorrelationFunction(mode=mode, edges=edges,
                                       data_positions1=positions,
                                       data_positions2=positions,
                                       engine='corrfunc',
                                       nthreads=nthreads,
                                       los='z',
                                       boxsize=boxsize)
    
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

# TODO write a fn for plotting power spectra/corr functions

# -----------
# MAIN
# -----------
def main():
    parser = argparse.ArgumentParser(description='Compute 2PCF from mock catalogs.')
    parser.add_argument('--variation', default='base',
                        help="which mock to use: 'base' or a sampled HOD variation id (e.g. 0007).")
    args = parser.parse_args()

    cfg = load_config()

    sim_type   = cfg['sim']['type']
    tracer     = cfg['tracer']
    seed_base  = cfg['seed_base']
    n_threads  = cfg.get('n_threads', 32)

    clust_cfg = cfg['clustering']
    corr_type = clust_cfg['corr_type']
    regime    = clust_cfg['regime']
    rsd       = clust_cfg['rsd']
    ells      = tuple(clust_cfg['ells'])
    n_seeds   = clust_cfg['n_seeds']
    pimax     = clust_cfg.get('pimax')
    fpar      = clust_cfg['ap']['fpar']
    fperp     = clust_cfg['ap']['fperp']

    variation    = args.variation
    variation_id = None if variation == 'base' else variation
    seed_vals    = [int(seed_base + i) for i in range(n_seeds)]

    # validate the requested variation exists in the HOD variation csv before doing any work
    if variation_id is not None:
        var_csv = hod_var_path(cfg['hod_variation']['name'])
        valid_ids  = {vid for vid, _ in load_hod_var(var_csv)}
        if variation_id not in valid_ids:
            raise ValueError(f'variation {variation_id!r} not in {var_csv}')

    edges, centers = _load_edges(corr_type, regime)
    xi_data = _output_meta(sim_type, variation, corr_type, regime, edges, centers, rsd, fpar)

    # output dir
    out_path = PATHS.xi(variation, corr_type, sim_type, regime, rsd, fpar)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'computing xi for variation={variation}')
    xi_start = time.time()

    xi_2d_vals, xi_1d_vals = [], []
    for seed in seed_vals:
        print(f'computing xi for seed={seed}')
        mock = load_mock(tracer, seed, sim_type, variation_id, rsd)

        xi_2d, xi_1d = mock2xi(mock=mock, corr_type=corr_type, edges=edges,
                                pimax=pimax, ells=ells, rsd=rsd,
                                fpar=fpar, fperp=fperp, nthreads=n_threads)
        xi_2d_vals.append(xi_2d)
        xi_1d_vals.append(xi_1d)

    xi_data['2d'] = np.asarray(xi_2d_vals)
    xi_data['1d'] = np.asarray(xi_1d_vals)

    print(f'took {time.time()-xi_start}s')
    print(f'saving the following dictionary @ {out_path}')
    print(xi_data)

    with open(out_path, 'wb') as f:
        pickle.dump(xi_data, f)

    print('done running script!')

if __name__ == '__main__':
    main()
