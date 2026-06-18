import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
from astropy.table import Table
import camb
import json
import time

from pycorr import TwoPointCorrelationFunction

## all useful functions for manipulating the data are here

## function to load the data into astropy tables
def get_mock(tracer,seed=101,param=None,param_value=None,sim_type='fiducial'):

    if param == None:
        data_folder = '/mnt/store2/araina/LRGmocks/base/'
        data_filename = '{}mock-base-seed={}-{}.fits'.format(tracer,seed,sim_type)
        return Table.read(data_folder+data_filename)

    else:
        data_folder = '/mnt/store2/araina/LRGmocks/{}/'.format(param)
        data_filename = '{}mock-{}={}-seed={}-{}.fits'.format(tracer,param,param_value,seed,sim_type)
        return Table.read(data_folder+data_filename)


## function to change positions to match corrfunc
def update_coords(table):
    
    boxsize = json.loads(table.meta['MOCK_INFO'])['boxsize']
    table['x'] = table['x'] + boxsize/2.
    table['y'] = table['y'] + boxsize/2.
    table['z'] = table['z'] + boxsize/2.

    return table


## function to add an RSD column to the data
def add_rsd(table,los='z'):

    mock_info = json.loads(table.meta['MOCK_INFO'])

    boxsize = mock_info['boxsize']
    Hz = mock_info['Hz']
    a = mock_info['a']

    if los == 'x': 
        x_rsd = table['x'] + table['vx']/(Hz*a)
    else:
        x_rsd = table['x']
        
    if los == 'y': 
        y_rsd = table['y'] + table['vy']/(Hz*a)
    else:
        y_rsd = table['y']
        
    if los == 'z': 
        z_rsd = table['z'] + table['vz']/(Hz*a)
    else:
        z_rsd = table['z']

    table['x_rsd'] = x_rsd % boxsize
    table['y_rsd'] = y_rsd % boxsize
    table['z_rsd'] = z_rsd % boxsize

    return table

## function to split table into centrals and satellites
def split_censat(table):

    Ncent = json.loads(table.meta['MOCK_INFO'])['Ncent']
    Nsat = json.loads(table.meta['MOCK_INFO'])['Nsat']

    print('splitting the mock into {} centralas and {} satellites'.format(Ncent,Nsat))
    
    cents = table[:Ncent]
    sats = table[Ncent:]

    return cents, sats


## function to make a cutout from the mock - assuming corrfunc coords (use after applying "update_coords")
def get_cutout(table,size,center):

    if min(table['x']) < 0.:
        print('need to first use "update_coords"')
        return
    
    boxsize = json.loads(table.meta['MOCK_INFO'])['boxsize']
    if ( ((center[0]-size[0]/2.) < 0) or ((center[0]+size[0]/2.) > boxsize) or
         ((center[1]-size[1]/2.) < 0) or ((center[1]+size[1]/2.) > boxsize) or
         ((center[2]-size[2]/2.) < 0) or ((center[2]+size[2]/2.) > boxsize) ):

        print('cutout is out of bounds')
        return

    else:

        mask = ( (table['x'] >= center[0]-size[0]/2.) & (table['x'] <= center[0]+size[0]/2.) & 
                 (table['y'] >= center[1]-size[1]/2.) & (table['y'] <= center[1]+size[1]/2.) & 
                 (table['z'] >= center[2]-size[2]/2.) & (table['z'] <= center[2]+size[2]/2.) )
        cutout = table[mask]
        
        return cutout

def paramvar_mocks(tracer='LRG',seed=101,param_dict={'base': [None]},sim_type='fiducial'):
    
    data_dict = {}
    
    for key in param_dict.keys():
        
        if key == 'base':
            param = None
        else:
            param = key
        
        mock_list = []
        
        for param_value in param_dict[key]:    
            mock = get_mock(tracer=tracer,seed=seed,param=param,param_value=param_value,sim_type=sim_type)
            mock = update_coords(mock)
            mock = add_rsd(mock,los='z')
            
            mock_list.append(mock)
        
        data_dict[key] = mock_list
            
    return data_dict

def mock2xi_rppi(mock,edges,pimax=None,rsd=True,fpar=1.,fperp=1.):
    
    if rsd:
        positions = np.asarray([mock['x_rsd']/fperp,
                                mock['y_rsd']/fperp,
                                mock['z_rsd']/fpar])
    else:
        positions = np.asarray([mock['x']/fperp,
                                mock['y']/fperp,
                                mock['z']/fpar])
        
    L = json.loads(mock.meta['MOCK_INFO'])['boxsize']
    boxsize = [L/fperp,L/fperp,L/fpar]
    
    start = time.time()
    result = TwoPointCorrelationFunction(mode='rppi', edges = edges,
                                         data_positions1 = positions,
                                         data_positions2 = positions,
                                         engine = 'corrfunc',
                                         nthreads = 32,
                                         los = 'z',
                                         boxsize = boxsize)
    
    xi_rppi_vals = result.get_corr()
    _, xi_wp_vals = result(pimax=pimax, return_sep=True)
    
    print('took {}s to compute xi'.format(time.time()-start))
    
    return xi_rppi_vals, xi_wp_vals

def mock2xi_smu(mock,edges,ells=(0,2,4),rsd=True,fpar=1.,fperp=1.):
    
    if rsd:
        positions = np.asarray([mock['x_rsd']/fperp,
                                mock['y_rsd']/fperp,
                                mock['z_rsd']/fpar])
    else:
        positions = np.asarray([mock['x']/fperp,
                                mock['y']/fperp,
                                mock['z']/fpar])
        
    L = json.loads(mock.meta['MOCK_INFO'])['boxsize']
    boxsize = [L/fperp,L/fperp,L/fpar]
    
    start = time.time()
    result = TwoPointCorrelationFunction(mode='smu', edges = edges,
                                         data_positions1 = positions,
                                         data_positions2 = positions,
                                         engine = 'corrfunc',
                                         nthreads = 32,
                                         los = 'z',
                                         boxsize = boxsize)
    
    xi_smu_vals = result.get_corr()
    _, xi_ell_vals = result(ells=ells, return_sep=True)
    
    print('took {}s to compute xi'.format(time.time()-start))
    
    return xi_smu_vals, xi_ell_vals

# def mock2xi_ell(mock,edges,ells=(0,2,4),rsd=True,fpar=1.,fperp=1.):
    
#     if rsd:
#         positions = np.asarray([mock['x_rsd']/fperp,
#                                 mock['y_rsd']/fperp,
#                                 mock['z_rsd']/fpar])
#     else:
#         positions = np.asarray([mock['x']/fperp,
#                                 mock['y']/fperp,
#                                 mock['z']/fpar])
    
#     L = json.loads(mock.meta['MOCK_INFO'])['boxsize']
#     boxsize = [L/fperp,L/fperp,L/fpar]
    
#     start = time.time()
#     result = TwoPointCorrelationFunction(mode='smu', edges = edges,
#                                          data_positions1 = positions,
#                                          data_positions2 = positions,
#                                          engine = 'corrfunc',
#                                          nthreads = 32,
#                                          los = 'z',
#                                          boxsize = boxsize)
    
#     xi_vals = result.get_corr()
#     xi_ells = result(ells=ells, return_sep=True)[1]
#     print('took {}s to compute xi'.format(time.time()-start))
    
#     return xi_ells

# def mock2xi_wp(mock,edges,pimax=None,rsd=True,fpar=1.,fperp=1.):
    
#     if rsd:
#         positions = np.asarray([mock['x_rsd']/fperp,
#                                 mock['y_rsd']/fperp,
#                                 mock['z_rsd']/fpar])
#     else:
#         positions = np.asarray([mock['x']/fperp,
#                                 mock['y']/fperp,
#                                 mock['z']/fpar])
    
#     L = json.loads(mock.meta['MOCK_INFO'])['boxsize']
#     boxsize = [L/fperp,L/fperp,L/fpar]
    
#     start = time.time()
#     result = TwoPointCorrelationFunction(mode='rppi', edges = edges,
#                                          data_positions1 = positions,
#                                          data_positions2 = positions,
#                                          engine = 'corrfunc',
#                                          nthreads = 32,
#                                          los = 'z',
#                                          boxsize = boxsize)
    
#     _, wp_vals = result(pimax=pimax, return_sep=True)
#     print('took {}s to compute xi'.format(time.time()-start))
    
#     return wp_vals

def ret_APparams(results_true,results_assumed,z):
    
    Hz_t = results_true.hubble_parameter(z) * u.km/u.s/u.Mpc
    Hz_a = results_assumed.hubble_parameter(z) * u.km/u.s/u.Mpc
    
    r_par_t = (const.c/Hz_t).to(u.Mpc)
    r_par_a = (const.c/Hz_a).to(u.Mpc)

    r_perp_t = results_true.comoving_radial_distance(z) * u.Mpc
    r_perp_a = results_assumed.comoving_radial_distance(z) * u.Mpc
    
    fpar = r_par_t/r_par_a
    fperp = r_perp_t/r_perp_a
    
    alpha_iso = (fpar*fperp**2)**(1/3)
    alpha_AP = fpar/fperp
    
    return fpar.value, fperp.value, alpha_iso.value, alpha_AP.value
