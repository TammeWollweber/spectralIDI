from functools import wraps
import numpy as np
import cupy as cp
from matplotlib import pyplot as plt
plt.ion()
import pyqtgraph as pg
import h5py as h5
from scipy import signal
from scipy import ndimage as ndi
import scipy.constants as const
from cupyx.scipy import signal as cusignal
from cupyx.scipy import ndimage as cundimage
from scipy.interpolate import interp1d
import glob
import sys
import os.path as op

dark = None
def load_dark():
    global dark
    with h5.File('/media/wittetam/Expansion/230103/dark_26_4000.h5', 'r') as f:
        dark = f['dark'][:]
    print(dark)

def radial_profile(data, center):
    x,y = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)
     
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile, r

def rad_2d(data, center):
    profile, r  = radial_profile(data, center)
    x = np.linspace(0, r.max(), len(profile))
    f = interp1d(x, profile)
    return f(r.ravel()).reshape(data.shape)

def rotate(corr, angle=12):
    return ndi.rotate(corr, angle, order=1, reshape=False)

def get_zscore(integ, corr, ucorr, ucorrsq, isums, cmask=None):
    noise = np.sqrt(ucorrsq-ucorr**2)
    cinteg = signal.fftconvolve(integ, integ[::-1,::-1], mode='same')
    #if cmask is None:
    #    cmask = cinteg > 20
    cmask = np.ones_like(corr).astype(int)
    zscore = (corr-np.nanmedian(corr[cmask])) / (noise/cinteg/isums.shape[0]**0.5)
    zscore[~cmask] = 0
    zscore[np.isnan(zscore) | np.isinf(zscore)] = 0
    return zscore


def get_zscore_1d(integ, corr, ucorr, ucorrsq, isums, cmask=None):
    noise = np.sqrt(ucorrsq-ucorr**2)
    cinteg = signal.fftconvolve(integ, integ[::-1,:], mode='full', axes=0)
    #if cmask is None:
    #    cmask = cinteg>20
    cmask = np.ones_like(corr).astype(int)
    zscore = (corr-np.nanmedian(corr[cmask])) / (noise/cinteg/isums.shape[0]**0.5)
    zscore[~cmask] = 0
    zscore[np.isnan(zscore) | np.isinf(zscore)] = 0
    return zscore


def load_file(exp='231127', oned=True, r=0):
    if oned:
        fname = 'data/{}_Run{}_1d.h5'.format(exp,r)
    else:
        fname = 'data/{}_Run{}.h5'.format(exp,r)
    with h5.File(fname, 'r') as f:
        integ = f['integ_frame'][:]
        corr = f['normalized_corr'][:]
        numr = f['corr_numr'][:]
        ucorrsq = f['corrsq_numr'][:]
        isums = f['frame_sums'][:]
    print('mean: ', np.nanmedian(corr))
    remove_nans(corr)
    if oned:
        zscore = get_zscore_1d(integ, corr, numr, ucorrsq, isums)
        return integ, corr-1, zscore, numr, isums
    else:
        zscore = get_zscore(integ, corr, numr, ucorrsq, isums)
        return integ, corr-1, zscore, numr, isums


def calc_diff(corr, theta=55, p_width=1):
    cen = np.array(corr.shape)//2
    c = np.copy(corr)
    c_theta = rotate(c, angle=theta)
    profile = c_theta[cen[0]-p_width:cen[0]+p_width+1, :]
    print(profile.shape)
    c[cen[0]-p_width:cen[0]+p_width+1, :] = profile
    c[:, cen[1]-p_width:cen[1]+p_width+1] = profile[:, (cen[1]-cen[0]):-(cen[1]-cen[0])].T

    rad_avg = rad_2d(c, cen)
    diff = c - rad_avg
    return diff, c, rad_avg

def calc_sum(dict_list):
    integ_sum = np.zeros_like(dict_list[0]['integ'])
    corr_sum = np.zeros_like(dict_list[0]['numr'])
    for i in range(len(dict_list)):
        corr_sum += dict_list[i]['numr'] 
        integ_sum += dict_list[i]['integ']
    corr_sum /= len(dict_list)
    integ_sum /= len(dict_list)
    cinteg = signal.fftconvolve(integ_sum, integ_sum[::-1,::-1])
    ncorr = corr_sum / cinteg
    remove_nans(ncorr)
    return ncorr, integ_sum

def remove_nans(corr):
    corr[np.isinf(corr) | np.isnan(corr)] = np.nanmedian(corr)
 

def cmod(corr):
    corr_new = np.copy(corr)
    corr_new[:,319] = np.nanmedian(corr_new)
    corr_new[269,:] = np.nanmedian(corr_new)
    return corr_new

def load_chunks(exp, run_num, num_chunks, angle=7.5, width=3):
    dicts = []
    for i in range(num_chunks):
        print('Chunk: ', i, end='\r')
        d = get_profile(exp, run_num, suffix='_{}_{}'.format(i*1000, (i+1)*1000), angle=angle, width=width)
        dicts.append(d)
    return dicts


def get_profile(exp, run_num, binned=False, suffix='', angle=7, cen=(1079,1279), width=3):
    if binned:
        suffix = '_binned' + suffix
    with h5.File('data/{}_Run{}{}.h5'.format(exp, run_num, suffix),'r') as f:
    #with h5.File('/home/wittetam/mount/{}_Run{}{}.h5'.format(exp, run_num, suffix),'r') as f:
        corr = f['normalized_corr'][:]
        if 'corrsq_numr' in f.keys():
            corrsq = f['corrsq_numr'][:]
        else:
            corrsq = None
        integ = f['integ_frame'][:]
        if 'cinteg' in f.keys():
            cinteg = f['cinteg'][:]
        else: 
            cinteg = None
        isums = f['frame_sums'][:]
        corr_numr = f['corr_numr'][:]
        cross_numr = f['cross_corr_numr'][:]
        bin_hist = f['bin_hist'][:]
        flist = np.array(f['file_list']).tolist().decode().split('\n')
    
    integ_corr = signal.fftconvolve(integ, integ[::-1, ::-1])
    ncorr = corr_numr / integ_corr
    remove_nans(ncorr)
    cross_norm = 1/2 * (cross_numr + cross_numr[::-1,::-1]) / integ_corr
    #corr = ncorr - cross_norm
    #remove_nans(corr)
    remove_nans(cross_norm)
    corr_rot = ndi.rotate(corr, angle, order=1, reshape=False)
    ncorr_rot = ndi.rotate(ncorr, angle, order=1, reshape=False)
    cross_rot = ndi.rotate(cross_norm, angle, order=1, reshape=False)
    line = corr_rot[:, cen[1]-width:cen[1]+width+1].mean(1)
    line_raw = ncorr_rot[:, cen[1]-width:cen[1]+width+1].mean(1)
    line_cross = cross_rot[:, cen[1]-width:cen[1]+width+1].mean(1)
    
    z_score = (ncorr - np.median(ncorr)) / (np.sqrt(len(isums)) * integ_corr**2 * (corrsq - corr_numr**2))

    z_score_rot = ndi.rotate(z_score, angle, order=1, reshape=False)
    z_score_line = z_score_rot[:, cen[1]-width:cen[1]+width+1].mean(1)
    my_dict = {}
    my_dict['corr'] = corr
    my_dict['raw'] = ncorr
    my_dict['cinteg'] = cinteg
    my_dict['isums'] = isums
    my_dict['integ'] = integ
    my_dict['cross'] = cross_norm
    my_dict['line'] = line
    my_dict['line_raw'] = line_raw
    my_dict['line_cross'] = line_cross
    my_dict['label'] = exp[-2:] + str(run_num)
    my_dict['z_score'] = z_score_rot
    my_dict['z_score_line'] = z_score_line
    my_dict['numr'] = corr_numr - 1/2* (cross_numr + cross_numr[::-1,::-1])
    my_dict['numr_raw'] = corr_numr
    my_dict['corrsq_numr'] = corrsq
    return my_dict
 
def get_3d_profile(exp, run_num, binned=False, suffix='', angle=7, cen=(1079,1279), width=3):
    if binned:
        suffix = '_binned' + suffix
    #with h5.File('data/{}_Run{}{}.h5'.format(exp, run_num, suffix),'r') as f:
    with h5.File('/mpsd/cni/processed/wittetam/{}_Run{}{}.h5'.format(exp, run_num, suffix),'r') as f:
        corr = f['normalized_corr'][:]
        #corrsq = f['corrsq_numr'][:]
        integ = f['integ_frame'][:]
        cinteg = f['cinteg'][:]
        isums = f['frame_sums'][:]
        corr_numr = f['corr_numr'][:]
        cross_numr = f['cross_corr_numr'][:]
       # bin_hist = f['bin_hist'][:]
       # flist = np.array(f['file_list']).tolist().decode().split('\n')
    
    ncorr = corr_numr / cinteg
    remove_nans(ncorr)
    cross_norm = 1/2 * (cross_numr + cross_numr[::-1,::-1]) / cinteg
    remove_nans(cross_norm)
    
    #z_score = (ncorr - np.median(ncorr)) / (np.sqrt(len(isums)) * cinteg**2 * (corrsq - corr_numr**2))

    my_dict = {}
    my_dict['corr'] = corr
    my_dict['raw'] = ncorr
    my_dict['cinteg'] = cinteg
    my_dict['isums'] = isums
    my_dict['integ'] = integ
    my_dict['cross'] = cross_norm
    my_dict['label'] = exp[-2:] + str(run_num)
    #my_dict['z_score'] = z_score
    my_dict['numr'] = corr_numr - 1/2* (cross_numr + cross_numr[::-1,::-1,::-1])
    my_dict['numr_raw'] = corr_numr
    my_dict['numr_cross'] = cross_numr
    #my_dict['corrsq_numr'] = corrsq
    return my_dict
 

def calc_sum_3d(dict_list, full=False):
    integ_sum = np.zeros_like(dict_list[0]['integ'])
    numr_sum = np.zeros_like(dict_list[0]['numr'])
    raw_sum = np.zeros_like(dict_list[0]['numr'])
    corr_sum = np.zeros_like(dict_list[0]['corr'])
    cross_sum = np.zeros_like(dict_list[0]['corr'])
    isums = []
    for i in range(len(dict_list)):
        numr_sum += dict_list[i]['numr'] 
        corr_sum += dict_list[i]['corr'] 
        integ_sum += dict_list[i]['integ']
        cross_sum += dict_list[i]['numr_cross']
        raw_sum += dict_list[i]['numr_raw']
        isums.extend(dict_list[i]['isums'])
    numr_sum /= len(dict_list)
    cross_sum /= len(dict_list)
    raw_sum /= len(dict_list)
    corr_sum /= len(dict_list)
    integ_sum /= len(dict_list)
    cinteg = calc_cinteg_3d(integ_sum)
    ncorr = numr_sum / cinteg
    remove_nans(ncorr)
    if full:
        return ncorr, corr_sum, integ_sum, cinteg, isums, cross_sum, raw_sum 
    else:
        return ncorr, corr_sum, integ_sum 

def calc_chunk(dict_list, fname, old=100, new=1000):
    for i in range(int(len(dict_list)//(new/old))):
        print(i, end='\r')
        ncorr_i, corr_i, integ_i, cinteg_i, isums_i, cross_i, raw_i = calc_sum_3d(dict_list[i*new//old:(i+1)*new//old], full=True)
        output_fname = fname + '_{}_{}_3d_333.h5'.format(i*1000, (i+1)*1000)
        print('Save to ', output_fname)
        with h5.File(output_fname, 'w') as f:
            f['normalized_corr'] = ncorr_i
            f['integ_frame'] = integ_i
            f['corr_numr'] = raw_i
            f['cross_corr_numr'] = cross_i
            f['frame_sums'] = isums_i
            f['cinteg'] = cinteg_i
            




def calc_cinteg_3d(integ):
    cinteg = cp.zeros((integ.shape[0]*2-1, integ.shape[1]*2-1,
                       integ.shape[2]*2-1), dtype='f8')
    coor = cp.array((np.where(integ != 0)), dtype='f4')
    npoints = coor.shape[1]
    num_blocks = npoints // 32 + 1
    num_threads = (npoints//num_blocks + 1,
                   npoints//num_blocks + 1)

    with open('raw_module.cu', 'r') as f:
            kernels = cp.RawModule(code=f.read())
    integ_kernel = kernels.get_function('cinteg_3d')
    integ_kernel((num_blocks,num_blocks), num_threads,
                      args=(cp.ascontiguousarray(coor[0,:]),
                            cp.ascontiguousarray(coor[1,:]),
                            cp.ascontiguousarray(coor[2,:]), npoints,
                            integ.shape[0], integ.shape[1],
                            integ.shape[2], 1., cp.array(integ,'f4'), cinteg))
    return cinteg.get()


def get_binned_profile(exp, run_num, binned=False, suffix='', angle=12, cen=(1079,1279), width=3):
    with h5.File('data/{}_Run{}{}_binned.h5'.format(exp, run_num, suffix),'r') as f:
        corr = f['normalized_corr'][:]
        corrsq = f['corrsq_numr'][:]
        integ = f['integ_frame'][:]
        isums = f['frame_sums'][:]
        corr_numr = f['corr_numr'][:]
        cross_numr = f['cross_corr_numr'][:]
        bin_hist = f['bin_hist'][:]
        flist = np.array(f['file_list']).tolist().decode().split('\n')
     
    remove_nans(corr)
    integ_corr = signal.fftconvolve(integ, integ[:,::-1,::-1], axes=(1,2)) * len(isums)**2 
    ncorr = (corr_numr / integ_corr).T * bin_hist
    ncorr[np.isnan(ncorr) | np.isinf(ncorr)] = 1
    weights = integ.mean((1,2))**2 * bin_hist
    wncorr = (ncorr * weights).T.sum(0) / weights.sum()
    ncorr = wncorr
    remove_nans(ncorr)
    cross_numr = 1/2 * (cross_numr + cross_numr[:, ::-1, ::-1])
    ncross = (cross_numr / integ_corr).T * bin_hist
    ncross[np.isnan(ncross) | np.isinf(ncross)] = 1
    weights = integ.mean((1,2))**2 * bin_hist
    wncross = (ncross * weights).T.sum(0) / weights.sum()
    ncross = wncross
    remove_nans(ncross)
    corr_rot = ndi.rotate(corr, angle, order=1, reshape=False)
    ncorr_rot = ndi.rotate(ncorr, angle, order=1, reshape=False)
    cross_rot = ndi.rotate(ncross, angle, order=1, reshape=False)
    line = corr_rot[:, cen[1]-width:cen[1]+width+1].mean(1)
    line_raw = ncorr_rot[:, cen[1]-width:cen[1]+width+1].mean(1)
    line_cross = cross_rot[:, cen[1]-width:cen[1]+width+1].mean(1)
    z_score = (ncorr - np.median(ncorr)) / (np.sqrt(len(isums)) * integ_corr**2 * (corrsq - corr_numr**2))
    z_score_rot = ndi.rotate(z_score, angle, order=1, reshape=False)
    z_score_line = z_score_rot[:, cen[1]-width:cen[1]+width+1].mean(1)
    my_dict = {}
    my_dict['corr'] = corr
    my_dict['raw'] = ncorr
    my_dict['isums'] = isums
    my_dict['integ'] = integ
    my_dict['cross'] = ncross
    my_dict['line'] = line
    my_dict['line_raw'] = line_raw
    my_dict['line_cross'] = line_cross
    my_dict['label'] = exp[-2:] + str(run_num)
    my_dict['z_score'] = z_score_rot
    my_dict['z_score_line'] = z_score_line
    my_dict['numr'] = corr_numr - 1/2* (cross_numr + cross_numr[::-1,::-1])
    my_dict['numr_raw'] = corr_numr
    my_dict['corrsq_numr'] = corrsq
    my_dict['bin_hist'] = bin_hist
    return my_dict


def info(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if not 'worker' in func.__name__:
            print(func.__name__)
        try:
            res = func(self, *args, **kwargs)
        finally:
            if not 'worker' in func.__name__:
                print('Done')
        return res
    return wrapper
        
def decorate_all(info):
    def decorator(cls):
        for name, obj in vars(cls).items():
            if callable(obj):
                try:
                    obj = obj.__func__
                except AttributeError:
                    pass
                setattr(cls, name, info(obj))
        return cls
    return decorator


