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
                 noise=60, efilter=False, det_dist=4, pixel_size=100):
        self.det_shape = tuple(np.array(det_shape) // binning)
        print('det_shape: ', self.det_shape)        
        self.binning = binning
        self.offset = self.det_shape[0]//2
        self.det_distance = det_dist 
        self.pixel_size = pixel_size*1e-6
        self.efilter = efilter
        self.num_photons = num_photons

        self.size_em1 = 7
        self.size_em2 = 10
        self.sample = None
        self.hits = []
        self.hit_size = None
        self.num_shots = num_shots 
        self.kvector = None
        self.r_k = None
        self.beat_period = 413 #attoseconds
        self.mode_period = 564 #attoseconds
        self.adu_phot = 160
        self.fft = None
        self.corr_list = []
        self.noise_level = noise
        self.background = np.round(10*self.num_photons).astype(int) #percentage of num_photons
        
        self.shots_per_file = 1000
        self.file_per_run_counter = 0
        self.data = []
        self.run_num = 0
        self.exp = None
        self.dir = '/mpsd/cni/processed/wittetam/sim/raw/'
        self._init_directory()
        self.num_cores = None
        self.integrated_signal = None

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
        #e_sep = 9000-8905 #main kbeta line but does not change with oxidation I assume
        e1 = 8975 #kbeta_2,5 Cu
        e2 = 8985 #kbeta_2,5 Cu1+
        e3 = 9000 #elastic 
        phi1 = 46.01
        phi2 = 45.95
        phi3 = 45.85
        e_sep = e3-e1
        e_center = np.round((e1+e3)/2).astype(int)
        self.pix_sep = self.det_distance * np.tan((phi1-phi3)*cp.pi/180) / self.pixel_size
        self.beta_shift = self.det_distance * np.tan((phi1-phi2)*cp.pi/180) / self.pixel_size
        self.e_res = (e_sep)/np.round(self.pix_sep)
        self.xkb1 = self.det_shape[1]//2 - self.pix_sep//2
        self.xkb2 = self.det_shape[1]//2 - self.pix_sep//2 + self.beta_shift
        self.xkel = self.det_shape[1]//2 + self.pix_sep//2
        e_range = np.round(self.det_shape[1] * self.e_res).astype(int)
        kvec = np.arange(self.det_shape[0])
        kvec -= self.det_shape[0]//2
        kscale_1d = 2 * np.pi / self.det_shape[0]
        kscale_corr = 2 * np.pi * (np.arange(self.det_shape[1])-self.det_shape[1]//2) * self.e_res/e_center
        kscale_2d = kscale_1d * kscale_corr + kscale_1d
        self.kvector = cp.outer(cp.array(kvec), cp.array(kscale_2d)).T
        self.sample = cp.array(self.sample)

    def create_sample(self):
        self.sample = np.zeros(self.det_shape)
        y,x = np.indices(self.det_shape)
        cen = np.array(self.det_shape)//2
        r = np.sqrt((x-cen[0])**2+(y-cen[1])**2)
        self.sample[r<self.size_em2] = 1
        self.sample[r<self.size_em1] += 1
        print('num_emitter 1: ', len(np.where(self.sample==1)[0]))
        print('num_emitter 2: ', len(np.where(self.sample==2)[0]))
        #self.sample[(self.offset-self.size_em2):(self.offset+self.size_em2),(self.offset-self.size_em2):(self.offset+self.size_em2)] = 1
        #self.sample[(self.offset-self.size_em1//2):(self.offset+self.size_em1//2),(self.offset-self.size_em1//2):(self.offset+self.size_em1//2)] += 1
            
    def lorentzian(self, x, a, x0, gam):
        gam = gam/2
        return a * gam**2 / (gam**2 + (x-x0)**2)

    def gaussian(self, x, a, mu, sig):
        sig = sig/2.355
        return a * cp.exp(-(x-mu)**2/(2*sig**2))
 
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
            diff_pattern = self._sim_frame(counter+n)
            if counter % num_jobs == 0:
                sys.stderr.write('\r%s: %d'%(counter, n))
            data[n] = diff_pattern
            n += 1
        
    def _sim_frame(self, counter):
        kbeta1 = self.lorentzian(cp.arange(self.det_shape[1]), 1, self.xkb1, 3.7/self.e_res)  
        kbeta2 = self.lorentzian(cp.arange(self.det_shape[1]), 1, self.xkb2, 3.7/self.e_res)  
        #elastic = self.lorentzian(cp.arange(self.det_shape[1]), self.xkel, 1, 3.7/self.e_res)
        elastic = cp.sqrt(self.gaussian(cp.arange(self.det_shape[1]), 0.1 ,self.xkel, 9/self.e_res))
        spectrum = kbeta1 + kbeta2
        
        pop = self.calc_beam_profile(counter)
        pop_max = cp.round(pop.max()).astype(int)
        num_modes = len(pop)

        diff_pattern = cp.zeros(self.det_shape)
        indices = cp.tile(cp.where(self.sample!=0)[0], (self.kvector.shape[0],1)).T
        ind1 = cp.tile(cp.where(self.sample==1)[0], (self.kvector.shape[0],1)).T
        ind2 = cp.tile(cp.where(self.sample==2)[0], (self.kvector.shape[0],1)).T

        r_k_el = cp.matmul(indices,self.kvector)/self.kvector.shape[1] #correct for broadcasting factor
        r_k1 = cp.matmul(ind1,self.kvector)/self.kvector.shape[1]
        r_k2 = cp.matmul(ind2,self.kvector)/self.kvector.shape[1]

        phases_fl1 = cp.array(cp.random.random(size=(1, num_modes, ind1.shape[0]))*2*cp.pi)
        phases_fl2 = cp.array(cp.random.random(size=(1, num_modes, ind2.shape[0]))*2*cp.pi)
        phases_el = cp.zeros((1, num_modes, indices.shape[0]))

        psi1_fl = cp.exp(1j*(r_k1[:,:,cp.newaxis].transpose(1,2,0)+phases_fl1)).sum(-1)
        psi2_fl = cp.exp(1j*(r_k2[:,:,cp.newaxis].transpose(1,2,0)+phases_fl2)).sum(-1)
        psi_el = cp.exp(1j*(r_k_el[:,:,cp.newaxis].transpose(1,2,0)+phases_el)).sum(-1)

        psi1_fl *= (pop / pop_max)
        psi2_fl *= (pop / pop_max)
        psi_el *= pop / pop_max

        int_fl1 = cp.abs(psi1_fl)**2 
        int_fl2 = cp.abs(psi2_fl)**2 
        int_el = cp.abs(psi_el)**2 
        int_el_norm = int_el * int_fl2.sum() / int_el.sum()

        int2d_fl1 = int_fl1.transpose(1,0)[:,:,cp.newaxis] * kbeta1[cp.newaxis,:]
        int2d_fl2 = int_fl2.transpose(1,0)[:,:,cp.newaxis] * kbeta2[cp.newaxis,:]
        int2d_el = int_el_norm.transpose(1,0)[:,:,cp.newaxis] * elastic[cp.newaxis,:]

        mode_int = int2d_fl1 + int2d_fl2 + int2d_el
        int_tot = mode_int.sum(0)
        int_tot /= int_tot.sum() / self.num_photons
        if self.efilter:
            int_filter = cundimage.gaussian_filter(int_tot, sigma=(0,1.13/self.e_res), mode='constant')
        else:
            int_filter = int_tot
        int_p = cp.random.poisson(cp.abs(int_filter),size=self.det_shape)
        int_p *= self.adu_phot
        diff_pattern += int_p
        #bg = cp.random.randint(0, diff_pattern.size, self.background)
        #diff_pattern.ravel()[bg] += self.adu_phot

        #gauss_noise = cp.random.normal(self.noise_level,2.5,self.det_shape)
        
        #diff_pattern += gauss_noise
        return diff_pattern

    def calc_beam_profile(self, counter):
        x = cp.arange(-1e4, 1e4, self.mode_period) #555 is 6.2/11 (FWHM/num_modes without polarization)
        y = self.num_photons * self.gaussian(x, 1, 0, 6.1*1000) #2596 is 6.1/2.35
        shot_noise = cp.random.uniform(-0.8, 0.8, len(y))
        y_noise = cp.round(y + shot_noise * y).astype(int)
        mask = cp.zeros_like(y_noise)
        mask[y_noise>=4] = 1 #at least 4 photons per mode, so 2 for each polarization
        y_final = (y_noise * mask)//2
        #cp.save('beam_profile_{}.npy'.format(counter), y_noise)
        y_final = y_final[y_final!=0]
        return cp.repeat(y_final,2) #repeat to take polarization into account


    def save_file(self, data, counter):
        dpath = self.dir + '{}/'.format(self.exp)
        np.ndarray.tofile((data.get()).astype('u2'), dpath+'Run{}_{:04d}.npy'.format(self.run_num,counter))
       

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
    efilter = config.getboolean(section, 'filter', fallback=True)
    det_dist = config.getint(section, 'det_dist', fallback=4)
    pixel_size = config.getint(section, 'pixel_size', fallback=100)

    det_shape = fshape
    #num_photons = np.ceil(args.photon_density * det_shape[0] * det_shape[1]).astype(int)
 
    sig = Signal(det_shape=det_shape, binning=binning, num_shots=num_shots, num_photons=num_photons, noise=noise, efilter=efilter, det_dist=det_dist, pixel_size=pixel_size)
    sig.create_sample()
    sig.sim_glob()

