import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.table import Table
import camb
from camb.dark_energy import DarkEnergyPPF
from pprint import pprint
import ast
import time
import json

import yaml
from abacusnbody.hod.abacus_hod import AbacusHOD

sim_type = 'fiducial'

# filename = 'LRG-config-fiducial.yaml' ## LRG-config-{}.yaml for "fiducial"
filename = 'LRG-config-{}.yaml'.format(sim_type)

path2config = '/home/araina/research/APproject/LRGmocks/config/' + filename
config = yaml.safe_load(open(path2config))

## obtain the sim params from the config file
sim_params = config['sim_params']
print('using the following Abacus sim params for the mocks')
pprint(sim_params)
print('\n')

## this function parses through the abacus.pars file
def get_abacus_pars(filename):
    
    abacus_pars = {}
    with open(filename) as f:
        lines = f.readlines()
        
        for line in lines[:-2]:
            line = line.strip()
            pairs = [x.strip() for x in line.split("=", 1)]
            
            key = pairs[0]
            val = pairs[1]
            
            try:
                abacus_pars[key] = ast.literal_eval(val)
            except Exception:
                abacus_pars[key] = val
    
    return abacus_pars

## this function scrapes the abacus_pars to get CAMB cosmology pars
def abacus_pars2camb_pars(abacus_pars):

    H0 = abacus_pars['H0']
    ombh2 = abacus_pars['omega_b']
    omch2 = abacus_pars['omega_cdm']
    omk = abacus_pars['Omega_K']
    tau = 0.06
    As = 2e-9
    ns = abacus_pars['n_s']
    thetastar = 0.0104109  # using Planck 2018 \theta_* value for fixing the CMB sound horizon (SH)
                           # this allows for computing the H0 using CAMB while keeping SH fixed

    w0 = abacus_pars['w0']
    wa = abacus_pars['wa']

    cosmo_info = {'H0' : H0,
                  'ombh2' : ombh2,
                  'omch2' : omch2,
                  'omk' : omk,
                  'tau' : tau,
                  'As' : As,
                  'ns' : ns,
                  'thetastar' : thetastar,
                  'w0' : w0,
                  'wa' : wa}

    camb_pars = camb.set_params(ombh2 = ombh2,
                                omch2 = omch2,
                                omk = omk,
                                tau = tau,
                                As = As,
                                ns = ns,
                                thetastar = thetastar)
    
    camb_pars.DarkEnergy = DarkEnergyPPF(w=w0,wa=wa)

    return camb_pars, cosmo_info


abacus_pars = get_abacus_pars(sim_params['sim_dir'] + 
                                '/' + sim_params['sim_name'] + 
                                '/' + 'abacus.par')

camb_pars, cosmo_info = abacus_pars2camb_pars(abacus_pars)
print('cosmology params for the mock [found in the header for fits file]')
pprint(cosmo_info)
print('\n')

cosmo = camb.get_results(camb_pars)

boxsize = abacus_pars['BoxSize']
z = sim_params['z_mock']
a = 1./(1.+sim_params['z_mock'])
H0 = cosmo.hubble_parameter(0.)
h0 = H0/100.
Hz = cosmo.hubble_parameter(z)
hz = Hz/100.


mock_info = {'boxsize': boxsize,
             'z': z,
             'a': a,
             'H0': H0,
             'h0': h0,
             'Hz': Hz,
             'hz': hz}

print('general info about the mocks [found in the header for fits file]')
pprint(mock_info)
print('\n')


## this function varies the HOD params given the config file
## this function is called in the loop for the param vals
def vary_HOD_param(config,tracer='LRG',param=None,param_val=None):
    
    if param==None:
        return config['HOD_params']

    else:

        import copy
        
        config_copy = copy.deepcopy(config)
        HOD_params_copy = config_copy['HOD_params']
        HOD_params_copy[tracer+'_params'][param]=param_val
        
        return HOD_params_copy


def mock2fits(mock_dict,tracer,filepath,mock_info,cosmo_info,HOD_info):

    cols = [mock_dict[tracer]['x'],
            mock_dict[tracer]['y'],
            mock_dict[tracer]['z'],
            mock_dict[tracer]['vx'],
            mock_dict[tracer]['vy'],
            mock_dict[tracer]['vz'],
            mock_dict[tracer]['mass']]
    col_names = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'm')

    t = Table(cols, names=col_names)

    t.meta['MOCK_INFO'] = json.dumps(mock_info)
    t.meta['COSMO_INFO'] = json.dumps(cosmo_info)
    t.meta['HOD_INFO'] = json.dumps(HOD_info)

    t.write(filepath, format='fits', overwrite=True)
    

    


Nmocks = 20                   ## number of mocks per a given set of params
tracer = 'LRG'                ## this is fixed for our purposes

tracer_param_name = 's_v'     ## for varying satellite velocities
# tracer_param_name = 'alpha_c'     ## for varying central velocities
# tracer_param_name = None

tracer_param_vals = [-1.0,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.0] # this is for "s_v"
# tracer_param_vals = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] # this is for "alpha_c"
# tracer_param_vals = [0.] # this is for the no variation case - make sure to choose a param that is 0 in baseline HOD


Nthread = 32
seed_vals = np.asarray([int(101+i) for i in range(Nmocks)])

for val,param_idx in zip(tracer_param_vals,range(len(tracer_param_vals))):

    HOD_params = vary_HOD_param(config = config,
                                tracer = tracer,
                                param = tracer_param_name,
                                param_val = val)

    print('---------- varying {} @ {} [{} of {}] ----------'.format(tracer_param_name,
                                                                    val,
                                                                    param_idx+1,
                                                                    len(tracer_param_vals)))
    print('using the following HOD params')
    pprint(HOD_params)
    print('\n')


    ## now we initialize the HOD Ball object for the given set of params
    
    print('initializing the HOD Ball object')
    init_start = time.time()
    Ball = AbacusHOD(sim_params,HOD_params)
    print('creating the HOD Ball took {}'.format(time.time()-init_start))
    print('\n')


    ## for a given ball object we run HOD to generate Nmocks # of mocks with synced seeds
    mocks_start = time.time()
    for sim_idx in range(Nmocks):

        print('-----------------------------------------------------')
        print('making mock {} of {}'.format(sim_idx+1,Nmocks))
        
        mock_dict = Ball.run_hod(Ball.tracers,
                             want_rsd=False,
                             Nthread=Nthread,
                             reseed=seed_vals[sim_idx],
                             verbose=True)
        
        ## adding central and satellite galxy info to the mock_info dictionary
        mock_info['Ncent'] = mock_dict[tracer]['Ncent']
        mock_info['Nsat'] = len(mock_dict[tracer]['x']) - mock_dict[tracer]['Ncent']

        if val != 0.:
            mock_folder = '/mnt/store2/araina/LRGmocks/{}/'.format(tracer_param_name)
            mock_filename = '{}mock-{}={}-seed={}-{}.fits'.format(tracer,
                                                          tracer_param_name,
                                                          val,
                                                          seed_vals[sim_idx],
                                                          sim_type)
        else:
            mock_folder = '/mnt/store2/araina/LRGmocks/base/'
            mock_filename = '{}mock-base-seed={}-{}.fits'.format(tracer,seed_vals[sim_idx],sim_type)
        
        mock2fits(mock_dict = mock_dict,
                  tracer = tracer,
                  filepath = mock_folder+mock_filename,
                  mock_info = mock_info,
                  cosmo_info = cosmo_info,
                  HOD_info = HOD_params[tracer+'_params'])

        print('mock saved @ {}'.format(mock_folder+mock_filename))
        print('-----------------------------------------------------')
        
    print('making {} mocks for {}={} took {}s'.format(Nmocks,tracer_param_name,val,time.time()-mocks_start))
    print('\n')

print('script ran succesfully!')
