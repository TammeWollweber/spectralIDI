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
from scipy import constants as const
from scipy import ndimage as ndi
import h5py as h5
import argparse
import configparser
import ctypes
import cupy as cp
from cupyx.scipy import signal as cusignal
from cupyx.scipy import ndimage as cundimage
plt.ion()

NUM_DEV = 1 
JOBS_PER_DEV = 1

class Signal():  
    def __init__(self, det_shape=(1024,1024), binning=8, num_shots=1000, num_photons=50,
                 emission_line='kb1', noise=60, efilter=False, det_dist=4, pixel_size=100):
        self.det_shape = tuple(np.array(det_shape) // binning)
        print('det_shape: ', self.det_shape)        
        self.binning = binning
        self.offset = self.det_shape[0]//2
        self.det_distance = det_dist 
        self.pixel_size = pixel_size*1e-6
        self.efilter = efilter
        self.num_photons = num_photons
        self.emission_line = emission_line

        self.size_em1 = 10
        self.size_em2 = 14
        self.sample_shape = (192,192)
        #self.size_np = 100e-9 #particle size in m
        self.sample = None
        self.hits = []
        self.hit_size = None
        self.num_shots = num_shots 
        self.kvector = None
        self.r_k = None
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

    def _init_sim(self, counter):
        if self.emission_line == 'kb1':
            e1 = 8905
            w1 = 3.7
            e2 = 8910
            w2 = 3.7
            phi1 = 46.48
            phi2 = 46.45
            darwin = 2.17
        elif self.emission_line == 'kb5':
            e1 = 8975
            e2 = 8985
            w1 = 3.7
            w2 = 3.7
            phi1 = 46.01
            phi2 = 45.95
            darwin = 2.13
        e3 = 9000 #elastic 
        phi3 = 45.85
        e_sep = e3-e1
        e_center = np.round((e1+e3)/2).astype(int)
        self.lam = self.pixel_size / self.det_distance
        fov = self.lam * self.sample.shape[0]
        kscale = fov
        self.pix_sep = self.det_distance * np.tan((phi1-phi3)*cp.pi/180) / self.pixel_size
        self.beta_shift = self.det_distance * np.tan((phi1-phi2)*cp.pi/180) / self.pixel_size
        self.e_res = (e_sep)/np.round(self.pix_sep)
        self.mode_period = (6200/np.ceil(6200/600)) * 2.17/self.e_res  #pulse duration over energy resolution compared to fabian times tc in attoseconds
        self.deltaE = self.det_distance * (cp.tan((phi1+darwin/3600)*np.pi/180) - cp.tan(phi1*np.pi/180)) / self.pixel_size #uncertainty from darwin plateau in units of pixel
        
        if counter == 0:
            print('bshift: ', self.beta_shift)
            print('eres: ', self.e_res)
            print('mode_period: ', self.mode_period)
            print('dE [eV], [pix]: ', self.deltaE*self.e_res, self.deltaE)
        self.xkb1 = self.det_shape[1]//2 - self.pix_sep//2
        self.xkb2 = self.det_shape[1]//2 - self.pix_sep//2 + self.beta_shift
        self.xkel = self.det_shape[1]//2 + self.pix_sep//2
        e_range = np.round(self.det_shape[1] * self.e_res).astype(int)
        kvec = np.arange(self.det_shape[0])
        kvec -= self.det_shape[0]//2
        kscale_corr = 2*np.pi*(np.arange(self.det_shape[1])-self.det_shape[1]//2) * self.e_res/e_center
        kscale_E = kscale * kscale_corr + kscale
        self.kvector = cp.outer(cp.array(kvec), cp.array(kscale_E)).T
        self.sample = cp.array(self.sample)

    def create_sample(self):
        self.sample = np.zeros(self.sample_shape)
        center = np.array(self.sample.shape)//2
        X, Y = np.meshgrid(np.arange(self.sample_shape[0])-center[0], np.arange(self.sample_shape[1])-center[1], indexing='ij')
        r = np.sqrt(X**2 + Y**2)
        self.sample[r<self.size_em2] = 1
        m_inner = np.zeros_like(self.sample)
        m_outer = np.zeros_like(self.sample)
        m_tot = np.zeros_like(self.sample)
        m_inner[r<self.size_em1] = 1
        m_outer[(r<self.size_em2) & (r>=self.size_em1)] = 1
        m_tot[self.sample!=0] = 1
        self.p_inner = (2*np.sqrt(np.abs(self.size_em1**2-r**2)*m_inner)).sum(-1)
        self.p_tot = (2*np.sqrt(np.abs(self.size_em2**2-r**2)*m_tot)).sum(-1)
        self.p_outer = self.p_tot - self.p_inner
        print('inner weight: ', self.p_inner.sum())
        print('outer weight: ', self.p_outer.sum())
                
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
            self._init_sim(i)
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
        elastic = cp.sqrt(self.gaussian(cp.arange(self.det_shape[1]), 1 ,self.xkel, 9/self.e_res))
        spectrum = kbeta1 + kbeta2
        
        pop = self.calc_beam_profile(counter)
        pop_max = cp.round(pop.max()).astype(int)
        num_modes = len(pop)
        if counter == 0:
            print('num modes: ', num_modes)

        diff_pattern = cp.zeros(self.det_shape)
        indices = cp.tile(cp.random.choice(cp.arange(0,self.sample.shape[0]), size=int(self.p_tot.sum()), p=self.p_tot/self.p_tot.sum()), (self.kvector.shape[0],1)).T
        ind_inner = cp.tile(cp.random.choice(cp.arange(0,self.sample.shape[0]), size=int(self.p_inner.sum()), p=self.p_inner/self.p_inner.sum()), (self.kvector.shape[0],1)).T
        ind_outer = cp.tile(cp.random.choice(cp.arange(0,self.sample.shape[0]), size=int(self.p_outer.sum()), p=self.p_outer/self.p_outer.sum()), (self.kvector.shape[0],1)).T

        r_k_el = cp.matmul(indices,self.kvector)/self.kvector.shape[1] #correct for broadcasting factor
        r_k1 = cp.matmul(ind_inner,self.kvector)/self.kvector.shape[1] # shape = (num_emitter, kshape[1])
        r_k2 = cp.matmul(ind_outer,self.kvector)/self.kvector.shape[1]

        phases_fl_inner = cp.array(cp.random.random(size=(1, num_modes, ind_inner.shape[0]))*2*cp.pi) #shape = (1, num_modes, num_emitter) 
        phases_fl_outer = cp.array(cp.random.random(size=(1, num_modes, ind_outer.shape[0]))*2*cp.pi)
        phases_el = cp.zeros((1, num_modes, indices.shape[0]))

        psi_fl_inner = cp.exp(1j*(r_k1[:,:,cp.newaxis].transpose(1,2,0)+phases_fl_inner)).sum(-1) # sum over all emitter
        psi_fl_outer = cp.exp(1j*(r_k2[:,:,cp.newaxis].transpose(1,2,0)+phases_fl_outer)).sum(-1)
        psi_el = cp.exp(1j*(r_k_el[:,:,cp.newaxis].transpose(1,2,0)+phases_el)).sum(-1)

        psi_fl_inner *= (pop / pop_max)
        psi_fl_outer *= (pop / pop_max)
        psi_el *= pop / pop_max

        int_fl_inner = cp.abs(psi_fl_inner)**2 
        int_fl_outer = cp.abs(psi_fl_outer)**2 
        int_el = cp.abs(psi_el)**2 

        int2d_fl_inner = (int_fl_inner.transpose(1,0)[:,:,cp.newaxis]).sum(0) * kbeta1[cp.newaxis,:] # sum over all modes
        int2d_fl_outer = (int_fl_outer.transpose(1,0)[:,:,cp.newaxis]).sum(0) * kbeta2[cp.newaxis,:]
        int2d_el = (int_el.transpose(1,0)[:,:,cp.newaxis]).sum(0) * elastic[cp.newaxis,:]

        mode_fl = int2d_fl_inner + int2d_fl_outer
        mode_fl *= self.num_photons / mode_fl.sum()
        int_tot = mode_fl + int2d_el * 0.1 * self.num_photons / int2d_el.sum()
        #int_tot = mode_fl + int2d_el * 10 * self.num_photons / int2d_el.sum()
        if self.efilter:
            int_filter = cundimage.gaussian_filter(int_tot, sigma=(0, self.deltaE), mode='constant')
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
        x = cp.arange(-1e4, 1e4, self.mode_period) #(FWHM/num_modes without polarization)
        y = self.num_photons * self.gaussian(x, 1, 0, 6.2*1000) #2596 is 6.2/2.35
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
    det_dist = float(config.get(section, 'det_dist', fallback=1))
    pixel_size = config.getint(section, 'pixel_size', fallback=100)
    line = config.get(section, 'emission_line', fallback='kb1')

    det_shape = fshape
    #num_photons = np.ceil(args.photon_density * det_shape[0] * det_shape[1]).astype(int)
 
    sig = Signal(det_shape=det_shape, binning=binning, num_shots=num_shots, num_photons=num_photons, emission_line=line, noise=noise, efilter=efilter, det_dist=det_dist, pixel_size=pixel_size)
    sig.create_sample()
    sig.sim_glob()

