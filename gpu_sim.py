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
import argparse
import configparser
import ctypes
import cupy as cp
from cupyx.scipy import signal as cusignal
from cupyx.scipy import ndimage as cundimage
plt.ion()

NUM_DEV = 3
JOBS_PER_DEV = 4

class Signal():  
    def __init__(self, det_shape=(1024,1024), binning=8, num_shots=1000, num_photons=50,
                 noise=60, num_modes=1, lines=True, incoherent=False, efilter=False, alpha_modes=2):
        self.det_shape = tuple(np.array(det_shape) // binning)
        print('det_shape: ', self.det_shape)        
        self.binning = binning
        self.lines = lines
        self.offset = self.det_shape[0]//2
        self.kscale_x = 2*np.pi/(self.det_shape[0]*4)
        self.ind_scale = 1000
        self.efilter = efilter
        self.num_modes = num_modes
        self.alpha_modes = alpha_modes

        self.size_emitter = 10
        self.sample = None
        self.center = np.array(self.det_shape)//2 - 1
        self.hits = []
        self.hit_size = None
        self.num_shots = num_shots 
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
        print('Simulate {} shots.'.format(self.num_shots))
        print('Save {} files.'.format(cp.ceil(self.num_shots/self.shots_per_file).astype(int)))

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
        num_files = np.ceil(self.num_shots / self.shots_per_file).astype(int)
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
        klist = [kalpha1, kalpha2]
        spectrum = kalpha1 + kalpha2
        diff_pattern = cp.zeros(self.det_shape)
        if self.alpha_modes == 1:
            num_scatterer = self.num_photons
            indices = cp.random.randint(0, self.ind_scale, size=(self.num_modes, self.num_photons//self.num_modes))
            if self.incoherent:
                phases_rand = cp.array(cp.random.random(size=(self.num_modes, num_scatterer//self.num_modes))*2*cp.pi)
            else:
                phases_rand = cp.zeros((self.num_modes, num_scatterer//self.num_modes))
            r_k = cp.matmul(self.loc_scatterer[indices%len(self.loc_scatterer),cp.newaxis],self.kvector[cp.newaxis,:])
            psi = cp.exp(1j*(r_k.transpose(2,0,1)+phases_rand)).sum(2).transpose(1,0)
            psi2d = psi[:,:,cp.newaxis] * spectrum[cp.newaxis,:]
            mode_int = cp.abs(psi2d)**2
            int_tot = mode_int.sum(0)
            int_tot /= int_tot.sum() / num_scatterer 

        elif self.alpha_modes == 2:
            num_scatterer = np.array([self.num_photons//3*2, self.num_photons//3])
            ind1 = cp.random.randint(0, self.ind_scale, size=(self.num_modes, num_scatterer[0]//self.num_modes))
            ind2 = cp.random.randint(0, self.ind_scale, size=(self.num_modes, num_scatterer[1]//self.num_modes))
            indices = (ind1, ind2)
            int_tot = cp.zeros(self.det_shape)
            for k in range(self.alpha_modes):
                if self.incoherent:
                    phases_rand = cp.array(cp.random.random(size=(self.num_modes, num_scatterer[k]//self.num_modes))*2*cp.pi)
                else:
                    phases_rand = cp.zeros(self.num_modes, num_scatterer[k]//self.num_modes)
                r_k = cp.matmul(self.loc_scatterer[indices[k]%len(self.loc_scatterer),cp.newaxis],self.kvector[cp.newaxis,:])

                psi = cp.exp(1j*(r_k.transpose(2,0,1)+phases_rand)).sum(2).transpose(1,0)
                psi2d = psi[:,:,cp.newaxis] * klist[k][cp.newaxis,:]
                mode_int = cp.abs(psi2d)**2
                int_tot_k = mode_int.sum(0)
                int_tot += int_tot_k
            int_tot /= int_tot.sum() / np.sum(num_scatterer)

        int_filter = cundimage.gaussian_filter(int_tot, sigma=(0,1.13//self.binning), mode='constant')
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
    parser = argparse.ArgumentParser(description='Simulate 1 spectral-IDI')
    parser.add_argument('-c', '--config_fname', help='Config file',
                        default='sim_config.ini')
    parser.add_argument('-s', '--config_section', help='Section in config file (default: sim)', default='sim')
    args = parser.parse_args()

    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read(args.config_fname)
    section = args.config_section

    fshape = tuple([int(i) for i in config.get(section, 'frame_shape').split()])
    binning = config.getint(section, 'binning', fallback=8)
    num_photons = config.getint(section, 'num_photons', fallback=1000)
    num_shots = config.getint(section, 'num_shots', fallback=1000)
    noise = config.getint(section, 'noise', fallback=60)
    modes= config.getint(section, 'modes', fallback=1)
    lines = config.getboolean(section, 'lines', fallback=False)
    incoherent = config.getboolean(section, 'incoherent', fallback=True)
    efilter = config.getboolean(section, 'filter', fallback=True)
    alpha = config.getint(section, 'alpha', fallback=2)

    det_shape = fshape
    #num_photons = np.ceil(args.photon_density * det_shape[0] * det_shape[1]).astype(int)
 
    sig = Signal(det_shape=det_shape, binning=binning, num_shots=num_shots, num_photons=num_photons, 
                 noise=noise, num_modes=modes, lines=lines, incoherent=incoherent,
                 efilter=efilter, alpha_modes=alpha)
    sig.create_sample()
    sig.sim_glob()

