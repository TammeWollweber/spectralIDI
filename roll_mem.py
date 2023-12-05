import sys
import os.path as op
import time
import glob
import configparser
import argparse
import multiprocessing as mp
import ctypes
import bisect

import numpy as np
import h5py
import cupy as cp
from cupyx.scipy import signal as cusignal
from cupyx.scipy import ndimage as cundimage
from scipy import signal
import natsort

NUM_DEV = 1
JOBS_PER_DEV = 1

class ProcessCorr():
    def __init__(self, flist, output_fname, mask_fname=None, fshape=(540, 640), start=0, end=None):
        self.fshape = fshape

        if mask_fname is None:
            self.np_mask = np.ones(self.fshape, dtype='f8')
        else:
            self.np_mask = np.load(mask_fname)

        with open('raw_module.cu', 'r') as f:
            kernels = cp.RawModule(code=f.read())
        self.k_thresh_frame = kernels.get_function('thresh_frame')

        self.flist = flist
        self.output_fname = output_fname
        self.np_dark = None
        self.corr = None
        self.integ = None
        self.isums = None
        self.isums_med = None
        self.isums_std = None
        self.start = start
        self.end = end
        self.min_val = None
        self.max_val = None
        self.mean_shot = None
        self.np_dark = np.zeros(self.fshape)
        self.frames_per_file = 1000
            
    def _init_corr(self, min_val=None, max_val=None):
        self.cudark = cp.array(self.np_dark)
        self.cumask = cp.array(self.np_mask)
        self.corr = cp.zeros((self.fshape[0], self.fshape[1], 2*self.fshape[1])).astype('f4')
        #self.corr = cp.zeros((self.fshape[0], self.fshape[1], self.fshape[1])).astype('f4')
        self.integ = cp.zeros((self.fshape[0], self.fshape[1])).astype('f4')
        #self.corrsq = cp.zeros_like(self.corr)

        self._init_flist(min_val, max_val)


    def _init_flist(self, min_val=None, max_val=None):
        if self.min_val is None and self.max_val is None:
            self.min_val = min_val
            self.max_val = max_val
        if self.end != -1: 
            self.flist = self.flist[self.start:self.end]
        else:
            self.flist = self.flist[self.start:]
        self.nframes = self.start*1000

    def proc_glob(self, min_val=None, max_val=None, **kwargs):
        self._init_flist(min_val=min_val, max_val=max_val)
        if len(self.flist) == 0:
            print('no files to process')
            return

        print('Processing %d files' % len(self.flist))
        num_jobs = NUM_DEV * JOBS_PER_DEV
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

        corr_arr = mp.Array(ctypes.c_float, num_jobs*self.fshape[0]*self.fshape[1]*self.fshape[1]*2)
        integ_arr = mp.Array(ctypes.c_float, num_jobs*int(np.product(self.fshape)))
        isums_arr = mp.Array(ctypes.c_float, 1000*len(self.flist))
        jobs = [mp.Process(target=self._mp_worker, args=(d, self.flist,
                                                         corr_arr,
                                                         integ_arr, isums_arr, 
                                                         kwargs['adu_thresh'],
                                                         kwargs['norm']))
                for d in range(num_jobs)]
        [j.start() for j in jobs]
        [j.join() for j in jobs]

        self.isums = np.frombuffer(isums_arr.get_obj(), dtype='f4')
        nframes = self.isums.shape[0]
        self.corr = np.frombuffer(corr_arr.get_obj(), dtype='f4').reshape((num_jobs,) + (self.fshape[0], self.fshape[1], 2*self.fshape[1]))
        self.corr = self.corr.sum(0) 
        self.integ = np.frombuffer(integ_arr.get_obj(), dtype='f4').reshape((num_jobs,) + self.fshape)
        self.integ = self.integ.sum(0)

    def _mp_worker(self, rank, flist, corr_arr, integ_arr, isums_arr, adu_thresh, norm):
        num_jobs = NUM_DEV * JOBS_PER_DEV
        devnum = rank // JOBS_PER_DEV
        cp.cuda.Device(devnum).use()
        self._init_corr()
        kwargs = {'adu_thresh': adu_thresh, 'norm': norm}
        mycorr = np.frombuffer(corr_arr.get_obj(), dtype='f4').reshape((num_jobs,) + (self.fshape[0], self.fshape[1], 2*self.fshape[1]))
        myinteg = np.frombuffer(integ_arr.get_obj(), dtype='f4').reshape((num_jobs,) + self.fshape)
    
        stime = time.time()
        for i, fname in enumerate(flist[rank::num_jobs]):
            self._proc_file(fname, i*num_jobs+rank, isums_arr, **kwargs)
            if rank == 0:
                sys.stderr.write(',  %.3f s/file\n' % ((time.time()-stime) / (i+1)))
        mycorr[rank] += self.corr.get()
        myinteg[rank] += self.integ.get()

    def _proc_file(self, fname, fnum, isums, **kwargs):
        if self.np_dark is None:
            print('Parse dark first')
            return
        num_jobs = NUM_DEV * JOBS_PER_DEV

        n = 0
        fptr = open(fname, 'rb')
        while True:
            fr = np.fromfile(fptr, '=u2', count=self.fshape[0]*self.fshape[1])
            if fr.size < self.fshape[0]*self.fshape[1]:
                break

            fr_corr, fr_integ = self._proc_frame(fr, **kwargs)
            isum_tmp = fr_integ.sum() #takes into account that different mask can be used now wrt to original isum data
            n_tot = fnum*1000 + n

            if n_tot % 1000 == 0: 
                isums[n_tot] = 0
                n += 1
                n_tot += 1
                continue

            if self.min_val is not None and self.max_val is not None:
                if float(fr_integ.sum()) < self.min_val or float(fr_integ.sum()) > self.max_val:
                    isums[n_tot] = 0
                    n += 1
                    n_tot += 1
                    continue

            isums[n_tot] = isum_tmp 

            self.corr += fr_corr
            self.integ += fr_integ
            if fnum % num_jobs == 0:
                sys.stderr.write('\r%s: %d'%(fname, n))
            n += 1
        fptr.close()

    def _proc_frame(self, frame_cpu, adu_thresh=120, norm=True):
        sfr = (cp.array(frame_cpu.reshape(self.fshape)) - self.cudark).astype('f4')

        npix = self.fshape[0] * self.fshape[1]
        bsize = npix // 32 + 1
        self.k_thresh_frame((bsize,), (32,), (sfr, npix, adu_thresh, self.cumask))
        if norm:
            cp.divide(sfr, cundimage.maximum_filter(sfr, 3, mode='constant'), out=sfr)
            sfr[cp.isnan(sfr) | cp.isinf(sfr)] = 0
        sfr_3d = cp.repeat(sfr[:,:,cp.newaxis], sfr.shape[1]*2, axis=2)
        sfr_3d = cp.pad(sfr_3d, ((0,0), (sfr.shape[1], sfr.shape[1]), (0,0)), mode='constant')
        #corr_3d = cp.zeros((sfr.shape[0], sfr.shape[1], sfr.shape[1])).astype('f4')
        sfr_shifted = self.shift_arr(sfr_3d)
        corr_3d = cusignal.fftconvolve(sfr_3d[:, self.fshape[1]:-self.fshape[1],:], sfr_shifted[::-1,self.fshape[1]:-self.fshape[1],:], mode='same', axes=0)
        #for i in range(-corr_3d.shape[-1]//2, corr_3d.shape[-1]//2):
        #    corr_3d[:,:,i+corr_3d.shape[-1]//2] = cusignal.fftconvolve(sfr, cundimage.shift(sfr, (0,i), mode='constant')[::-1,:], mode='same', axes=0)

        return corr_3d, sfr

    def shift_arr(self,a):
        ridx, cidx, zidx = cp.ogrid[:a.shape[0], :a.shape[1], :1]
        zidx = cp.ones(a.shape[2])[cp.newaxis, cp.newaxis, :].astype(int) * a.shape[2]//2
        shifts = cp.arange(a.shape[2], 2*a.shape[2])
        cidx = (cidx - shifts[cp.newaxis, cp.newaxis, :])
        a_shifted = a[ridx, cidx, zidx]
        return a_shifted

    def norm_corr(self):
        integ = cp.array(self.integ)
        integ_3d = cp.repeat(integ[:,:,cp.newaxis], integ.shape[1]*2, axis=2).astype('f4')
        integ_3d = cp.pad(integ_3d, ((0,0), (integ.shape[1],integ.shape[1]), (0,0)), mode='constant')
        integ_shifted = self.shift_arr(integ_3d)
        integ_3d = cusignal.fftconvolve(integ_3d[:, self.fshape[1]:-self.fshape[1], :], integ_shifted[::-1, self.fshape[1]:-self.fshape[1], :], mode='same', axes=0) 
        #for i in range(-integ.shape[-1]//2, integ.shape[-1]//2):
        #    cinteg[:,:,i+integ.shape[-1]//2] = cusignal.fftconvolve(integ, cundimage.shift(integ, (0,i))[::-1,:], mode='same', axes=0)
        ncorr = self.corr / integ_3d.get()
        return ncorr

    def save_corr(self, idx=None):
        if self.corr is None:
            print('Nothing to save')
            return

        if idx is not None:
            self.output_fname = self.output_fname[:-3] + '_' + str(idx*self.frames_per_file) + '.h5'
        
        print('Writing output to', self.output_fname)
        with h5py.File(self.output_fname, 'w') as f:
            f['corr_numr'] = self.corr / len(self.isums)
            f['integ_frame'] = self.integ / len(self.isums)
            #f['corrsq_numr'] = self.corrsq[0] / len(self.isums)
            f['frame_sums'] = self.isums
            f['normalized_corr'] = self.norm_corr()
            f['goodpix_mask'] = self.np_mask
            f['file_list'] = '\n'.join(self.flist)

def main():
    parser = argparse.ArgumentParser(description='Correlate dense frames')
    parser.add_argument('-c', '--config_fname', help='Config file',
                        default='config.ini')
    parser.add_argument('-s', '--config_section', help='Section in config file (default: corr)', default='corr')
    parser.add_argument('-r', '--run_num', type=int, help='run_num, if None take from config file', default=-1)
    args = parser.parse_args()

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(args.config_fname)
    section = args.config_section

    fshape = tuple([int(i) for i in config.get(section, 'frame_shape').split()])
    mask_fname = config.get(section, 'mask_fname', fallback=None)
    exp = config.get(section, 'date_str')
    if args.run_num == -1:
        runs = [int(r) for r in config.get(section, 'runs').split()]
    else:
        runs = [args.run_num]
    #start = config.getint(section, 'start_block', fallback=0)
    starts = [int(s) for s in config.get(section, 'start_block').split()]
    file_chunk = config.getint(section, 'file_chunk', fallback=1000)
    end_orig = config.getint(section, 'stop_block', fallback=None)
    norm = config.getboolean(section, 'do_norm', fallback=False)
    threshold = config.getfloat(section, 'adu_threshold', fallback=300)
    isums_fname = config.get(section, 'isums_fname', fallback=None)
    min_val = config.getint(section, 'min_val', fallback=None)
    max_val = config.getint(section, 'max_val', fallback=None)
    suffix = config.get(section, 'output_suffix', fallback=None)



    for r in runs:
        for s in starts:
            print('Processing run %s:%d' % (exp, r))
            data_glob = '/mpsd/cni/processed/wittetam/sim/raw/%s/Run%d_*.npy' % (exp, r)
            output_fname = 'data/%s_Run%d' % (exp,r)
            start = s
            if end_orig is None:
                end = s+file_chunk
            else:
                end = end_orig
            if end is None:
                end = -1
            output_fname += '_{}_{}'.format(start, end)
            output_fname += '_roll.h5'
            
            print('Writing output to', output_fname)

            flist = natsort.natsorted(glob.glob(data_glob))
            pc = ProcessCorr(flist, output_fname, mask_fname, fshape, start, end)
            pc.proc_glob(min_val, max_val, norm=norm, adu_thresh=threshold)
            pc.save_corr()

if __name__ == '__main__':
    main()
