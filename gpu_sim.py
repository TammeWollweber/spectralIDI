# -*- coding: utf-8 -*-
"""
Created on Thu May  2 13:44:31 2019

@author: DerMäuser
"""

import sys
import numpy as np
import time
import glob
from matplotlib import pyplot as plt
from datetime import datetime
import os
import multiprocessing as mp
from scipy import signal
from scipy import ndimage as ndi
import h5py as h5
import ctypes
import cupy as cp
from cupyx.scipy import signal as cusignal
from cupyx.scipy import ndimage as cundimage
import argparse
plt.ion()

NUM_DEV = 3
JOBS_PER_DEV = 4

class Signal():  
    def __init__(self, det_shape=(1024,1024), binning=8, shots=1000, num_photons=50,
                 noise=60, num_modes=1, lines=True, incoherent=False, efilter=False,
                 total=False, alpha_modes=2):
        self.det_shape = tuple(np.array(det_shape) // binning)
        print('det_shape: ', self.det_shape)        
        self.binning = binning
        self.lines = lines
        self.offset = self.det_shape[0]//2
        self.kscale_x = 2*np.pi/(self.det_shape[0]*4)
        self.efilter = efilter
        self.full_frame = total
        self.num_modes = num_modes
        self.alpha_modes = alpha_modes

        self.size_emitter = 32
        self.sample = None
        self.center = np.array(self.det_shape)//2 - 1
        self.hits = []
        self.hit_size = None
        self.shots = shots 
        self.num_scatterer = None
        self.loc_scatterer = None
        self.kvector = None
        self.r_k = None
        
        self.adu_phot = 160
        self.fft = None
        self.corr_list = []
        self.noise_level = noise
        
        self.shots_per_file = 1000
        self.file_per_run_counter = 0
        self.data = []
        self.run_num = 0
        self.exp = None
        self.dir = '/mpsd/cni/processed/wittetam/sim/raw/'
        self._init_directory()
        self.num_photons = num_photons
        self.num_cores = None
        self.integrated_signal = None

        self.incoherent = incoherent

    def _init_directory(self):
        self.exp = datetime.today().strftime('%y%m%d')
        dpath = self.dir + self.exp 
        if not os.path.exists(dpath):
            os.makedirs(dpath)
        else:
            flist = glob.glob(dpath+'/*.npy')
            flist = [os.path.basename(f) for f in flist]
            flist = list(set([f.split('_')[0] for f in flist]))
            flist.sort()
            self.run_num = len(flist)
            print(flist)
        print('Start simulation run {}.'.format(self.run_num))
        print('Simulate {} shots.'.format(self.shots))
        print('Save {} files.'.format(cp.ceil(self.shots/self.shots_per_file).astype(int)))

    def _init_sim(self):
        self.loc_scatterer = cp.nonzero(cp.array(self.sample))[0] - cp.array(self.center[0])
        #min_val = cp.min(cp.nonzero(self.sample)[0] - self.center[0])
        #max_val = cp.max(cp.nonzero(self.sample)[0] - self.center[0])
        #self.loc_scatterer = cp.linspace(min_val, max_val, self.num_photons)
        self.kvector = np.arange(self.det_shape[0])
        self.kvector -= self.det_shape[0]//2
        self.kvector = self.kvector * self.kscale_x
        self.kvector = cp.array(self.kvector)

    def create_sample(self):
        self.sample = cp.zeros(self.det_shape)
        if self.lines:
            self.sample[30:40, 30:80] = 1
            self.sample[50:60, 30:80] = 1
            self.sample[70:80, 30:80] = 1
        else:
            half_size = int(self.size_emitter/2)
            self.sample[(self.offset-half_size):(self.offset+half_size),(self.offset-half_size):(self.offset+half_size)] = 1
            
    def lorentzian(self, x, x0, a, gam):
        return a * gam**2 / (gam**2 + (x-x0)**2)
 
    def sim_glob(self):
        self.file_per_run_counter = 0
        num_jobs = NUM_DEV * JOBS_PER_DEV
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

        jobs = [mp.Process(target=self.worker, args=(d,)) for
                    d in range(num_jobs)]
        [j.start() for j in jobs]
        [j.join() for j in jobs]
        self.run_num += 1

      
    def worker(self, rank):
        num_jobs = NUM_DEV * JOBS_PER_DEV
        devnum = rank // JOBS_PER_DEV
        cp.cuda.Device(devnum).use()

        data_arr = mp.Array(ctypes.c_double, self.shots_per_file)
        num_files = np.ceil(self.shots / self.shots_per_file).astype(int)
        stime = time.time()
        for i, _ in enumerate(np.arange(num_files)[rank::num_jobs]):
            idx = i*num_jobs+rank
            cp.random.seed(idx+int(time.time()))
            self._init_sim()
            data = cp.zeros((self.shots_per_file, self.det_shape[0], self.det_shape[1]))
            self.sim_file(data, idx)
            self.save_file(data, idx)
            if rank == 0:
                sys.stderr.write(',  %.3f s/file\n' % ((time.time()-stime) / (i+1)))

    def sim_file(self, data, counter):
        num_jobs = NUM_DEV * JOBS_PER_DEV
        n = 0
        while n < self.shots_per_file:
            diff_pattern = self._sim_frame()
            if counter % num_jobs == 0:
                sys.stderr.write('\r%s: %d'%(counter, n))
            data[n] = diff_pattern
            n += 1
        
    def _sim_frame(self):
        kx1 = (1024//2+16)//self.binning
        kx2 = (1024//2-17)//self.binning
        kalpha1 = self.lorentzian(cp.arange(self.det_shape[1]), kx1, 1, 2.11/self.binning)  
        kalpha2 = self.lorentzian(cp.arange(self.det_shape[1]), kx2, 0.5, 2.17/self.binning)  
        num_scatterer = np.array([self.num_photons//3*2, self.num_photons//3])
        
        diff_pattern = cp.zeros(self.det_shape)
        #indices = cp.tile(cp.arange(self.num_photons)//self.num_modes, self.num_modes).reshape(self.num_modes, self.num_photons)
        indices = cp.arange(self.num_photons)
        cp.random.shuffle(indices)
        indices = (indices[:num_scatterer[0]], indices[-num_scatterer[1]:])
        alpha_indices = []
        for i in range(2):
            alpha_indices.append(cp.array(indices[i][:len(indices[i])//self.num_modes*self.num_modes]).reshape(self.num_modes, len(indices[i])//self.num_modes))
        spec_mode_conv = cp.zeros(self.det_shape)
        for k in range(self.alpha_modes):
            if self.full_frame:
                if self.incoherent:
                    phases_rand = cp.array(cp.random.random(size=(self.num_modes, num_scatterer[k]//self.num_modes, self.det_shape[1]))*2*cp.pi)
                else:
                    phases_rand = cp.zeros((self.num_modes, num_scatterer[k]//self.num_modes, self.det_shape[1]))

            else:
                if self.incoherent:
                    phases_rand = cp.array(cp.random.random(size=(self.num_modes, num_scatterer[k]//self.num_modes))*2*cp.pi)
                else:
                    phases_rand = cp.zeros(self.num_modes, num_scatterer[k]//self.num_modes)
            r_k = cp.matmul(self.loc_scatterer[alpha_indices[k]%len(self.loc_scatterer),cp.newaxis],self.kvector[cp.newaxis,:])

            if self.full_frame:
                psi = cp.exp(1j*(r_k[:,:,:,np.newaxis].transpose(2,0,1,3)+phases_rand)).sum(2).transpose(1,0,2)
            else:
                psi = cp.exp(1j*(r_k.transpose(2,0,1)+phases_rand)).sum(2).transpose(1,0)
            mode_int = cp.abs(psi)**2
            int_tot = mode_int.sum(0)
            
            spec_mode = cp.zeros(self.det_shape)
            if k == 0:
                if self.full_frame:
                    spec_mode = int_tot
                else:
                    spec_mode[:,kx1] = int_tot
                if self.efilter:
                    mode_conv = cusignal.fftconvolve(spec_mode, kalpha1[cp.newaxis,:], axes=1, mode='same')
                else:
                    mode_conv = spec_mode
                spec_mode_conv += mode_conv / mode_conv.sum() * num_scatterer[k]
            else:
                if self.full_frame:
                    spec_mode = int_tot
                else:
                    spec_mode[:,kx2] = int_tot
                if self.efilter:
                    mode_conv = cusignal.fftconvolve(spec_mode, kalpha2[cp.newaxis,:], axes=1, mode='same')
                else:
                     mode_conv = spec_mode
                spec_mode_conv += mode_conv / mode_conv.sum() * num_scatterer[k]
            
            int_filter = cundimage.gaussian_filter(spec_mode_conv, sigma=(0,1.13//self.binning), mode='constant')
            #int_filter = cundimage.gaussian_filter(spec_mode_conv, sigma=(0,0), mode='constant')
            int_p = cp.random.poisson(cp.abs(int_filter),size=self.det_shape)
            int_p *= self.adu_phot
            diff_pattern += int_p

        gauss_noise = cp.random.normal(self.noise_level,2.5,self.det_shape)
        diff_pattern += gauss_noise
        return diff_pattern

    def save_file(self, data, counter):
        dpath = self.dir + '{}/'.format(self.exp)
        np.ndarray.tofile((data.get()/100).astype('u2'), dpath+'Run{}_{:04d}.npy'.format(self.run_num,counter))
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', nargs='+', type=int, help='det_shape',
                        default=(1024, 1024))
    parser.add_argument('-b', '--binning', type=int, default=8, help='binning 1024x1024 det')
    parser.add_argument('-N', '--num_shots', type=int, default=1000)
    parser.add_argument('-M', '--photon_density', type=float, default=0.03, help='Number of photons per pixel')
    parser.add_argument('-o', '--noise', type=int, default=60, help='Noise level')
    parser.add_argument('-n', '--num_photons', type=int, default=1000, help='Number of photons per shot')
    parser.add_argument('-m', '--modes', type=int, default=30, help='Number of modes')
    parser.add_argument('-l', '--lines', type=int, default=0, help='Sample shape')
    parser.add_argument('-i', '--incoherent', type=int, default=1, help='Incoherent/coherent simulation')
    parser.add_argument('-f', '--filter', type=int, default=0, help='Convolve energies')
    parser.add_argument('-t', '--total', type=int, default=0, help='Populate full (total) detector')
    parser.add_argument('-a', '--alpha', type=int, default=2, help='Number of kalpha modes')

    args = parser.parse_args()

    det_shape = tuple(args.size)
    #num_photons = np.ceil(args.photon_density * det_shape[0] * det_shape[1]).astype(int)
    num_photons = args.num_photons
 
    sig = Signal(det_shape=det_shape, binning=args.binning, shots=args.num_shots, num_photons=num_photons, 
                 noise=args.noise, num_modes=args.modes, lines=args.lines, incoherent=args.incoherent,
                 efilter=args.filter, total=args.total, alpha_modes=args.alpha)
    sig.create_sample()
    sig.sim_glob()

