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
    def __init__(self, flist, output_fname, mask_fname=None, fshape=(540, 640), num_bins=1):
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
        self.start = 0
        self.end = None
        self.min_val = None
        self.max_val = None
        self.mean_shot = None
        self.num_bins = num_bins
        self.bin_boundaries = np.empty(self.num_bins)
        self.frames_per_file = 32
        self.np_dark = np.zeros(self.fshape)

    def _init_corr(self, min_val=None, max_val=None):
        self.cudark = cp.array(self.np_dark)
        self.cumask = cp.array(self.np_mask)
        self.corr = cp.zeros((self.num_bins, self.fshape[0], 2*self.fshape[1], self.fshape[1]))
        self.integ = cp.zeros((self.num_bins, self.fshape[0], self.fshape[1]))
        self.corrsq = cp.zeros_like(self.corr)
        self.bin_hist = cp.zeros(self.num_bins)

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

        corr_arr = mp.Array(ctypes.c_double, num_jobs*self.num_bins*self.fshape[0]*self.fshape[1]*2*self.fshape[1])
        integ_arr = mp.Array(ctypes.c_double, num_jobs*self.num_bins*int(np.product(self.fshape)))
        corrsq_arr = mp.Array(ctypes.c_double, num_jobs*self.num_bins*self.fshape[0]*self.fshape[1]*2*self.fshape[1])
        bin_hist_arr = mp.Array(ctypes.c_double, num_jobs*self.num_bins)
        isums_arr = mp.Array(ctypes.c_double, 1000*len(self.flist))
        jobs = [mp.Process(target=self._mp_worker, args=(d, self.flist,
                                                         corr_arr,
                                                         integ_arr, corrsq_arr, isums_arr, 
                                                         bin_hist_arr,
                                                         kwargs['adu_thresh'],
                                                         kwargs['norm']))
                for d in range(num_jobs)]
        [j.start() for j in jobs]
        [j.join() for j in jobs]

        self.isums = np.frombuffer(isums_arr.get_obj())
        nframes = self.isums.shape[0]
        self.corr = np.frombuffer(corr_arr.get_obj()).reshape((num_jobs, self.num_bins) + (self.fshape[0], self.fshape[1]*2, self.fshape[1]))
        self.corr = self.corr.sum(0) 
        self.integ = np.frombuffer(integ_arr.get_obj()).reshape((num_jobs, self.num_bins) + self.fshape)
        self.integ = self.integ.sum(0)
        self.corrsq = np.frombuffer(corrsq_arr.get_obj()).reshape((num_jobs,) + self.corr.shape)
        self.corrsq = self.corrsq.sum(0) 
        self.bin_hist = np.frombuffer(bin_hist_arr.get_obj()).reshape((num_jobs, self.num_bins)).sum(0)

    def _mp_worker(self, rank, flist, corr_arr, integ_arr, corrsq_arr, isums_arr, bin_hist, adu_thresh, norm):
        num_jobs = NUM_DEV * JOBS_PER_DEV
        devnum = rank // JOBS_PER_DEV
        cp.cuda.Device(devnum).use()
        self._init_corr()
        kwargs = {'adu_thresh': adu_thresh, 'norm': norm}
        mycorr = np.frombuffer(corr_arr.get_obj()).reshape((num_jobs, self.num_bins) + (self.fshape[0], self.fshape[1]*2, self.fshape[1]))
        myinteg = np.frombuffer(integ_arr.get_obj()).reshape((num_jobs,self.num_bins) + self.fshape)
        mycorrsq = np.frombuffer(corrsq_arr.get_obj()).reshape((num_jobs,) + mycorr.shape[1:])
    
        mybin_hist = np.frombuffer(bin_hist.get_obj()).reshape((num_jobs, self.num_bins))

        stime = time.time()
        #counter = 1
        
        for i, fname in enumerate(flist[rank::num_jobs]):
            self._proc_file(fname, i*num_jobs+rank, isums_arr, **kwargs)
            if rank == 0:
                sys.stderr.write(',  %.3f s/file\n' % ((time.time()-stime) / (i+1)))
        mycorr[rank] += self.corr.get()
        myinteg[rank] += self.integ.get()
        mycorrsq[rank] += self.corrsq.get()
        mybin_hist[rank] += self.bin_hist.get()
            #if ((i+1) * num_jobs) % self.frames_per_file:
            #    self.save_corr(idx=counter)
            #    counter += 1

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
            bin_idx = None
            if self.num_bins == 1:
                bin_idx = 0
            else:
                bin_idx = cp.argmin(cp.abs(cp.array(self.bin_boundaries) - isum_tmp))


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

            self.bin_hist[bin_idx] += 1
            self.corr[bin_idx] += fr_corr
            self.integ[bin_idx] += fr_integ
            self.corrsq[bin_idx] += cp.square(fr_corr)
            if fnum % num_jobs == 0:
                sys.stderr.write('\r%s: %d'%(fname, n))
            n += 1
        fptr.close()

    def _proc_frame(self, frame_cpu, adu_thresh=120, norm=True):
        sfr = cp.array(frame_cpu.reshape(self.fshape)) - self.cudark

        npix = self.fshape[0] * self.fshape[1]
        bsize = npix // 32 + 1
        self.k_thresh_frame((bsize,), (32,), (sfr, npix, adu_thresh, self.cumask))
        if norm:
            cp.divide(sfr, cundimage.maximum_filter(sfr, 3, mode='constant'), out=sfr)
            sfr[cp.isnan(sfr) | cp.isinf(sfr)] = 0
        sfr_3d = cp.repeat(sfr[:,:,cp.newaxis], sfr.shape[1], axis=2)
        sfr_pad = cp.pad(sfr_3d, ((0,0), (sfr.shape[1]//2, sfr.shape[1]//2), (0,0)), mode='constant')
        #corr_3d = cp.zeros((sfr.shape[0], sfr.shape[1]*2, sfr.shape[1]))
        sfr_shifted = self.shift_arr(sfr_pad)
        corr_3d = cusignal.fftconvolve(sfr_pad, sfr_shifted[::-1,:,:], mode='same', axes=0)
        #for i in range(-corr_3d.shape[-1]//2, corr_3d.shape[-1]//2):
        #    corr_3d[:,:,i+corr_3d.shape[-1]//2] = cusignal.fftconvolve(sfr_pad, cundimage.shift(sfr_pad, (0,i), mode='constant')[::-1,:], mode='same', axes=0)

        return corr_3d, sfr

    def shift_arr(self,a):
        ridx, cidx, zidx = cp.ogrid[:a.shape[0], :a.shape[1], :a.shape[2]]
        shifts = cp.arange(-a.shape[2]//2, a.shape[2]//2)
        shifts[shifts<0] += a.shape[1]
        cidx_new = (cidx - shifts[cp.newaxis, cp.newaxis, :])
        a_shifted = a[ridx, cidx_new, zidx]
        return a_shifted

    def norm_corr(self):
        integ = cp.array(self.integ[0])
        integ_3d = cp.repeat(integ[:,:,cp.newaxis], integ.shape[1], axis=2)
        integ_pad = cp.pad(integ_3d, ((0,0), (integ.shape[1]//2,integ.shape[1]//2), (0,0)), mode='constant')
        integ_shifted = self.shift_arr(integ_pad)
        cinteg = cusignal.fftconvolve(integ_pad, integ_shifted[::-1,:,:], mode='same', axes=0)
        #for i in range(-cinteg.shape[-1]//2, cinteg.shape[-1]//2):
        #    cinteg[:,:,i+cinteg.shape[-1]//2] = cusignal.fftconvolve(integ_pad, cundimage.shift(integ_pad, (0,i))[::-1,:], mode='same', axes=0)
        ncorr = self.corr[0] / cinteg.get()
        return ncorr

    def save_corr(self, idx=None):
        if self.corr is None:
            print('Nothing to save')
            return

        print(self.bin_hist)
        if idx is not None:
            self.output_fname = self.output_fname[:-3] + '_' + str(idx*self.frames_per_file) + '.h5'
        
        print('Writing output to', self.output_fname)
        with h5py.File(self.output_fname, 'w') as f:
            f['corr_numr'] = self.corr[0] / len(self.isums)
            f['integ_frame'] = self.integ[0] / len(self.isums)
            f['corrsq_numr'] = self.corrsq[0] / len(self.isums)
            f['frame_sums'] = self.isums
            f['normalized_corr'] = self.norm_corr()
            f['goodpix_mask'] = self.np_mask
            f['file_list'] = '\n'.join(self.flist)
            f['bin_hist'] = self.bin_hist

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
    num_bins = config.getint(section, 'num_bins', fallback=1)
    min_val = config.getint(section, 'min_val', fallback=None)
    max_val = config.getint(section, 'max_val', fallback=None)
    suffix = config.get(section, 'output_suffix', fallback=None)



    for r in runs:
        for s in starts:
            print('Processing run %s:%d' % (exp, r))
            data_glob = '/mpsd/cni/processed/wittetam/sim/raw/%s/Run%d_*.npy' % (exp, r)
            output_fname = 'data/%s_Run%d' % (exp,r)
            output_fname += '_roll.h5'
            print('Writing output to', output_fname)

            flist = natsort.natsorted(glob.glob(data_glob))
            pc = ProcessCorr(flist, output_fname, mask_fname, fshape, num_bins)
            pc.proc_glob(min_val, max_val, norm=norm, adu_thresh=threshold)
            pc.save_corr()

if __name__ == '__main__':
    main()
