# scripts/analysis/clustering.py
# two-point corr functions from generated mock catalogs
# pass --mode to select HOD variation

# -----------
# IMPORTS
# -----------
import argparse
import pickle
import time

import numpy as np

from src.utils import PROJECT_ROOT, load_config, get_param_dict, load_mock, mock2xi


# -----------
# FUNCTIONS
# -----------
# helpers
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

def _output_path(mode, corr_type, sim_type, regime, rsd, fpar):
    ap = (fpar != 1)
    tags = [tag for tag, active in (('rsd', rsd), ('AP', ap)) if active]
    suffix = '_' + '+'.join(tags) if tags else ''
    filename = f'xi_{corr_type}_{mode}_{sim_type}_{regime}{suffix}.pkl'
    return PROJECT_ROOT / 'results' / 'xi' / f'xi_{mode}' / filename

def _load_edges(corr_type, regime):
    bins_filename = f'{corr_type}_bins_{regime}.npz'
    bin_data = np.load(PROJECT_ROOT / 'data' / 'bins' / bins_filename)
    edges   = (bin_data['edges0'],   bin_data['edges1'])
    centers = (bin_data['centers0'], bin_data['centers1'])
    return edges, centers


# -----------
# MAIN
# -----------
def main():
    parser = argparse.ArgumentParser(description='Compute 2PCF from mock catalogs.')
    parser.add_argument('--mode', required=True,
                        help='HOD variation mode to compute (e.g. base, alpha_c, s_v).')
    args = parser.parse_args()

    cfg = load_config()

    sim_type   = cfg['sim']['type']
    tracer     = cfg['tracer']
    param_dict = get_param_dict(cfg)
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

    mode      = args.mode
    seed_vals = [int(seed_base + i) for i in range(n_seeds)]

    if mode not in param_dict:
        raise ValueError(f'HOD mode {mode!r} not supported :( — available: {list(param_dict.keys())}')

    edges, centers = _load_edges(corr_type, regime)
    xi_data = _output_meta(sim_type, mode, corr_type, regime, edges, centers, rsd, fpar)

    ## auto-create output dir
    output_path = _output_path(mode, corr_type, sim_type, regime, rsd, fpar)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f'computing xi for mode={mode}')
    xi_start = time.time()

    for param_value in param_dict[mode]:
        base   = param_value is None
        param  = None if base else mode
        prefix = ''   if base else f'{mode}={param_value}_'
        label  = mode if base else f'{mode}={param_value}'

        print(f'starting with {label}')
        param_start = time.time()

        xi_2d_vals, xi_1d_vals = [], []
        for seed in seed_vals:
            print(f'computing xi for seed={seed}')
            mock = load_mock(tracer, seed, param, param_value, sim_type, rsd)

            xi_2d, xi_1d = mock2xi(mock=mock, corr_type=corr_type, edges=edges,
                                    pimax=pimax, ells=ells, rsd=rsd,
                                    fpar=fpar, fperp=fperp, nthreads=n_threads)
            xi_2d_vals.append(xi_2d)
            xi_1d_vals.append(xi_1d)

        xi_data[f'{prefix}2d'] = np.asarray(xi_2d_vals)
        xi_data[f'{prefix}1d'] = np.asarray(xi_1d_vals)
        print(f'{label} took {time.time()-param_start}s')

    print(f'took {time.time()-xi_start}s')
    print(f'saving the following dictionary @ {output_path}')
    print(xi_data)

    with open(output_path, 'wb') as f:
        pickle.dump(xi_data, f)

    print('done running script!')


if __name__ == '__main__':
    main()
