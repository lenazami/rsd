# scripts/gen/mocks.py
# generates galaxy mock catalogs w/ abacushod
# makes n mocks for fixed fiducial hods (tracer default: LRG).

# -----------
# IMPORTS
# -----------

# -----------
# FUNCTIONS
# -----------


# -----------
# MAIN
# -----------
def main():
    parser = argparse.ArgumentParser(description='generate galaxy mock catalogs w/ abacushod')
    parser.add_argument('-n', '--n-mocks', type=int, default=1,
                        help='number of mocks to generate for a given hod, default=1')
    parser.add_argument('-t', '--tracer', type=str, default='LRG',
                        help='choose a tracer to generate a mock: LRG (default), ELG, QSO')
    parser.add_argument('-pf', '--prep-force', type=bool, default=False,
                        help='forces a prep sims as subsamples to generate mocks.')
    args = parser.parse_args()

    script_start = time.perf_counter()
    # load from config
    cfg  = load_config()
    combos = list(cfg_combos(cfg.get('sim_id', {})))
    
    for over in combos:
        combo_start = time.perf_counter()
        sim = SimID.from_cfg(cfg, **over)
        mock = replace(MockID.from_cfg(cfg), sim=sim)  # combos override sim_id
        n_threads = cfg.get('n_threads', 32)
        seeds     = [mock.seed + i for i in range(args.n_mocks)]
        
        # if not all subsamples exist, this will prep the sim, otherwise will just return 
        # if args.prep flag is active, it will prep subsamples regardless of completion
        _ = prep_sim(sim, mock, cfg, prep_force=args.prep_force)
        
        sim_params = sim.sim_params()
        hod_params = mock.hod_params(sim.z)
        hod_pars = hod_params[mock.param_key]
        
        print('mocks using the following abacus sim params:')
        pprint(sim_params)
        print('using the following hod params:')
        pprint(hod_pars)
        print()

        mock_info = setup_cosmology(sim)

        init_start = time.perf_counter()
        print('initializing the HOD Ball object')
        Ball = AbacusHOD(sim_params, hod_params)
        print(f'creating the HOD Ball took {time.perf_counter() - init_start:.3f}\n')

        # TODO: will eventually need to figure out a conditional for handling hod variations
        # mock gen loop
        mocks_start = time.time()
        for i, seed in enumerate(seeds):
            print('-'*50)
            print(f'making mock {i+1} of {len(seeds)} (seed={seed})')
            path = gen_cat(Ball, replace(mock, seed=seed), mock_info,
                        hod_pars, n_threads=n_threads)
            print(f'mock saved @ {path}')
        print('-'*50)
        print(f'made {len(seeds)} mocks for {mock.tracer} h{mock.hod_id} in {time.time()-mocks_start:.3f}s')
        if len(combos)>1:
            print(f'combo took {time.perf_counter()-combo_start:.5f}s')
    print(f'script ran successfully! {time.perf_counter()-script_start:.5f}s')

if __name__ == '__main__':
    main()
