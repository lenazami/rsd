import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
from astropy.table import Table
from matplotlib import cm, colors

# import camb
from utils import *

import json
import pprint
import time
import pickle

## script params - need to chnage for param var, sim_type, range of scales
sim_type = 'fiducial'
tracer = 'LRG'
param_dict = {'base': [None],
              'alpha_c': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
              's_v': [-1.0,-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8,1.0]}
seed_vals = [101,102,103,104,105]

corr_type = 'smu'
mode = 's_v'
regime = 'small_scales'
rsd = True
# refer to the short script called 'get_AP.py' to obtain these for AP variation
# fpar = 1.
fpar = 0.947
# fperp = 1.
fperp = 0.962
pimax = None
ells = (0,2,4)


bins_filename = '{}_bins_{}.npz'.format(corr_type,regime)
bin_data = np.load('data/'+ bins_filename)
edges = (bin_data['edges0'],bin_data['edges1'])
centers = (bin_data['centers0'],bin_data['centers1'])

if mode == 'base':
    
    print('computing xi for the base HOD')
    
    xi_data = {}
    xi_2d_vals = []
    xi_1d_vals = []
    
    xi_start = time.time()
    
    for seed in seed_vals:
        
        print('computing xi for seed={}'.format(seed))
        
        mock = get_mock(tracer=tracer,
                        seed=seed,
                        param=None,
                        param_value=None,
                        sim_type=sim_type)
        
        mock = update_coords(mock)
        
        if rsd:
            mock = add_rsd(mock, los='z')
        
        if corr_type == 'rppi':
            xi_2d, xi_1d = mock2xi_rppi(mock=mock,
                                          edges=edges,
                                          pimax=pimax,
                                          rsd=rsd,
                                          fpar=fpar,
                                          fperp=fperp)
        
        elif corr_type == 'smu':
            xi_2d, xi_1d = mock2xi_smu(mock=mock,
                                       edges=edges,
                                       ells=ells,
                                       rsd=rsd,
                                       fpar=fpar,
                                       fperp=fperp)
        
        
        else:
            print('corr_type not supported :(')
            continue
        
        xi_2d_vals.append(xi_2d)
        xi_1d_vals.append(xi_1d)
    
    print('took {}s for all seeds'.format(time.time()-xi_start))

    xi_data['sim_type'] = sim_type
    xi_data['var_HOD_param'] = mode
    xi_data['corr_type'] = corr_type
    xi_data['bin_type'] = regime
    
    xi_data['2d'] = np.asarray(xi_2d_vals)
    xi_data['1d'] = np.asarray(xi_1d_vals)
    xi_data['edges'] = edges
    xi_data['centers'] = centers
    
    if (rsd == True) and (fpar !=1):
        xi_data['rsd'] = True
        xi_data['AP'] = True
        xi_filename = 'xi_{}_{}_{}_{}_rsd+AP.pkl'.format(corr_type,mode,sim_type,regime)
        
    elif (rsd == True) and (fpar == 1):
        xi_data['rsd'] = True
        xi_data['AP'] = False
        xi_filename = 'xi_{}_{}_{}_{}_rsd.pkl'.format(corr_type,mode,sim_type,regime)
        
    elif (rsd == False) and (fpar != 1):
        xi_data['rsd'] = False
        xi_data['AP'] = True
        xi_filename = 'xi_{}_{}_{}_{}_AP.pkl'.format(corr_type,mode,sim_type,regime)
    
    else:
        xi_data['rsd'] = False
        xi_data['AP'] = False
        xi_filename = 'xi_{}_{}_{}_{}.pkl'.format(corr_type,mode,sim_type,regime)
        
    print('saving the following dictionary @ data/xi_base/')
    print(xi_data)
    
    with open('data/xi_base/' + xi_filename, 'wb') as f:
        pickle.dump(xi_data, f)


elif mode != 'base':
    
    print('computing xi for varying {} parameter'.format(mode))
    xi_data = {}
    xi_start = time.time()
    
    for param_value in param_dict[mode]:
        
        xi_2d_vals = []
        xi_1d_vals = []
        
        print('starting with {}={}'.format(mode,param_value))     
        param_start = time.time()
        
        for seed in seed_vals:
            
            print('computing xi for seed={}'.format(seed))
            
            mock = get_mock(tracer=tracer,
                            param=mode,
                            seed=seed,
                            param_value=param_value,
                            sim_type=sim_type)
        
            mock = update_coords(mock)
            
            if rsd:
                mock = add_rsd(mock, los='z')
                
            if corr_type == 'rppi':
                xi_2d, xi_1d = mock2xi_rppi(mock=mock,
                                           edges=edges,
                                           pimax=pimax,
                                           rsd=rsd,
                                           fpar=fpar,
                                           fperp=fperp)
        
            elif corr_type == 'smu':
                xi_2d, xi_1d = mock2xi_smu(mock=mock,
                                           edges=edges,
                                           ells=ells,
                                           rsd=rsd,
                                           fpar=fpar,
                                           fperp=fperp)
            
            else:
                print('corr_type not supported :(')
                continue
            
            xi_2d_vals.append(xi_2d)
            xi_1d_vals.append(xi_1d)
            
        print('{}={} took {}s'.format(mode,param_value,time.time()-param_start))
        
        xi_data['{}={}_2d'.format(mode,param_value)] = np.asarray(xi_2d_vals)
        xi_data['{}={}_1d'.format(mode,param_value)] = np.asarray(xi_1d_vals)
        
    print('took {}s'.format(time.time()-xi_start))
    
    xi_data['sim_type'] = sim_type
    xi_data['var_HOD_param'] = mode
    xi_data['corr_type'] = corr_type
    xi_data['bin_type'] = regime
    
    xi_data['edges'] = edges
    xi_data['centers'] = centers
    
    if (rsd == True) and (fpar !=1):
        xi_data['rsd'] = True
        xi_data['AP'] = True
        xi_filename = 'xi_{}_{}_{}_{}_rsd+AP.pkl'.format(corr_type,mode,sim_type,regime)
        
    elif (rsd == True) and (fpar == 1):
        xi_data['rsd'] = True
        xi_data['AP'] = False
        xi_filename = 'xi_{}_{}_{}_{}_rsd.pkl'.format(corr_type,mode,sim_type,regime)
        
    elif (rsd == False) and (fpar != 1):
        xi_data['rsd'] = False
        xi_data['AP'] = True
        xi_filename = 'xi_{}_{}_{}_{}_AP.pkl'.format(corr_type,mode,sim_type,regime)
    
    else:
        xi_data['rsd'] = False
        xi_data['AP'] = False
        xi_filename = 'xi_{}_{}_{}_{}.pkl'.format(corr_type,mode,sim_type,regime)
        
    print('saving the following dictionary @ data/xi_base/')
    print(xi_data)
    
    with open('data/xi_{}/'.format(mode) + xi_filename, 'wb') as f:
        pickle.dump(xi_data, f)
            

else:
    print('HOD mode not supported :(')


print('done running script!')
    
                
            
        
        
        
        
        
        
        
