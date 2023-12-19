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
from configobj import ConfigObj
import ctypes
import cupy as cp
from cupyx.scipy import signal as cusignal
from cupyx.scipy import ndimage as cundimage
plt.ion()

NUM_DEV = 1 
JOBS_PER_DEV = 1

class Signal():  
    def __init__(self, elements, lines, det_shape=(1024,1024), binning=8, num_shots=1000, num_photons=50,
                 emission_line='kb1', noise=60, efilter=False, det_dist=4, si_dist=30e-2, pixel_size=100, particle_size=350):

        self.elements = elements
        self.lines = lines
        self.specs = []

        self.det_shape = tuple(np.array(det_shape) // binning)
        print('det_shape: ', self.det_shape)        
        self.binning = binning
        self.offset = self.det_shape[0]//2
        self.det_distance = det_dist 
        self.si_dist = si_dist
        self.det_dist_E = self.det_distance - self.si_dist
        self.pixel_size = pixel_size*1e-6
        self.efilter = efilter
        self.num_photons = num_photons
        self.emission_line = emission_line

        self.size_em1 = 21
        self.size_em2 = 30
        self.sample_shape = (64,64)
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
        self.num_cores = None
        self.integrated_signal = None
        self.tau = None
        self.width = None
        self.pulse_dur = 6200
        self.lam = 0.19e-9
        self.particle_size = particle_size*1e-9
        self.rpix_size = None
        self.rscale = None
        self._init_directory()
        self._init_lines()

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

    def _init_lines(self):
        config = ConfigObj('elements.ini')
        for e in self.elements:
            for l in self.lines:
                print(e,l)
                self.specs.append(config[e][l])
        self.specs.append(config[e]['elastic'])
        self.tau = int(config[e]['tau'])
        self.width = float(config[e]['kalpha2']['w'])
        print('coherence time, width kalpha2: ', self.tau, self.width)


    def _init_sim(self, counter):
        phi1 = float(self.specs[0]['phi'])
        phi2 = float(self.specs[1]['phi'])
        E1 = float(self.specs[0]['E'])
        E2 = float(self.specs[1]['E'])
        darwin_b1 = float(self.specs[0]['darwin'])
        darwin_b2 = float(self.specs[1]['darwin'])
    
        E3 = float(self.specs[-1]['E']) #elastic 
        phi3 = float(self.specs[-1]['phi'])
        e_sep = E3 - E1
        e_center = np.round((E1+E3)/2).astype(int)
        phi_cen = (phi1+phi3)/2

        N = self.det_distance / self.pixel_size
        self.rpix_size = self.particle_size / (self.size_em2*2)
        self.rscale = (self.rpix_size/self.lam) / N

        print('det dist E: ', self.det_dist_E)
        x1 = self.det_dist_E * np.tan(np.abs(phi1-phi_cen)*cp.pi/180)
        x2 = self.det_dist_E * np.tan(np.abs(phi2-phi_cen)*cp.pi/180)
        x3 = self.det_dist_E * np.tan(np.abs(phi3-phi_cen)*cp.pi/180)
        self.pix_sep = np.abs(x1+x3) / self.pixel_size
        self.beta_shift =  np.abs(x1-x2) / self.pixel_size
 
        self.e_res = (e_sep)/np.round(self.pix_sep)
        #self.deltaE = self.det_distance * (cp.tan((phi1+darwin/3600)*np.pi/180) - cp.tan(phi1*np.pi/180)) / self.pixel_size #uncertainty from darwin plateau in units of pixel

        self.dE_b1 = np.ceil(2*self.det_dist_E * (np.tan((phi1+darwin_b1/3600-phi_cen)*np.pi/180) - np.tan((phi1-phi_cen)*np.pi/180)) / self.pixel_size).astype(int) #uncertainty from darwin plateau in units of pixel
        self.dE_b2 = np.ceil(2*self.det_dist_E * (np.tan((phi2+darwin_b2/3600-phi_cen)*np.pi/180) - np.tan((phi2-phi_cen)*np.pi/180)) / self.pixel_size).astype(int) #uncertainty from darwin plateau in units of pixel
        self.deltaE = np.max((self.dE_b1, self.dE_b2))

        self.mode_period = self.tau * self.width / (self.deltaE * self.e_res)
    
        if counter == 0:
            print('phi: ', phi1, phi2, phi3, phi_cen)
            print('rscale: ', self.rscale)
            print('pix_sep: ', self.pix_sep)
            print('bshift: ', self.beta_shift)
            print('Energy resolution from pixels: ', self.e_res)
            print('Uncertainty from darwin: ', self.dE_b1, self.dE_b2)
            print('ecenter: ', e_center)
            print('mode_period: ', self.mode_period)
            print('dE [eV], [pix]: ', self.deltaE*self.e_res, self.deltaE)
        self.xkb1 = self.det_shape[1]//2 - self.pix_sep//2
        self.xkb2 = self.det_shape[1]//2 - self.pix_sep//2 + self.beta_shift
        self.xkel = self.det_shape[1]//2 + self.pix_sep//2
        print('xpos: ', self.xkb1, self.xkb2, self.xkel)
        e_range = np.round(self.det_shape[1] * self.e_res).astype(int)
        self.kvector = cp.arange(self.det_shape[0]) - self.det_shape[0]//2
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
        self.p_inner = (2*np.sqrt(np.abs(self.size_em1**2-r**2)*m_inner)).sum(-1).astype('f4')
        self.p_tot = (2*np.sqrt(np.abs(self.size_em2**2-r**2)*m_tot)).sum(-1).astype('f4')
        self.p_outer = self.p_tot - self.p_inner
                
    def lorentzian(self, x, a, x0, gam):
        gam = gam/2
        return a * gam**2 / (gam**2 + (x-x0)**2)

    def gaussian(self, x, a, mu, sig):
        sig = sig/2.355
        return a * cp.exp(-(x-mu)**2/(2*sig**2))
 
    def darwin_kernel(self, width):
        x0 = int(self.det_shape[1]//2)
        y = cp.zeros(self.det_shape[1])
        y[int(x0-width//2):int(x0+width//2+1)] = 1
        xupper = cp.arange(1, self.det_shape[1]-(x0+width//2-1))
        xlower = cp.arange(1, x0-width//2+1) * -1
        yupper = (xupper - cp.sqrt(xupper**2-1))**2
        ylower = (xlower + cp.sqrt(xlower**2-1))**2
        y[int(x0+width//2):] = yupper
        y[:int(x0-width//2)] = ylower[::-1]
        return y


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
        kbeta1 = self.lorentzian(cp.arange(self.det_shape[1]), 1, self.xkb1, 2.97/self.e_res)  
        kbeta2 = self.lorentzian(cp.arange(self.det_shape[1]), 1, self.xkb2, 2.97/self.e_res)  
        elastic = cp.sqrt(self.gaussian(cp.arange(self.det_shape[1]), 0.1 ,self.xkel, 9/self.e_res))
        spectrum = kbeta1 + kbeta2
        
        pop = self.calc_beam_profile(counter)
        pop_max = cp.round(pop.max()).astype(int)
        num_modes = len(pop)
        if counter == 0:
            print('num modes: ', num_modes)

        inner_weight = self.p_inner.sum()
        outer_weight = self.p_outer.sum()

        diff_pattern = cp.zeros(self.det_shape)
        indices = cp.random.choice(cp.arange(0,self.sample.shape[0]), size=int(1000), p=self.p_tot/self.p_tot.sum()).astype('u2')
        ind_inner = cp.random.choice(cp.arange(0,self.sample.shape[0]), size=np.round(1000*(inner_weight/(inner_weight+outer_weight))).astype(int), p=self.p_inner/self.p_inner.sum()).astype('u2')
        ind_outer = cp.random.choice(cp.arange(0,self.sample.shape[0]), size=np.round(1000*(outer_weight/(inner_weight+outer_weight))).astype(int), p=self.p_outer/self.p_outer.sum()).astype('u2')

        r_k_el = cp.outer(indices*self.rscale,self.kvector)
        r_k1 = cp.outer(ind_inner*self.rscale,self.kvector)
        r_k2 = cp.outer(ind_outer*self.rscale,self.kvector)
        phases_fl_inner = 2*cp.pi*cp.array(cp.random.random(size=(num_modes, ind_inner.shape[0]))).astype('f4')

        phases_fl_outer = 2*cp.pi*cp.array(cp.random.random(size=(num_modes, ind_outer.shape[0]))).astype('f4')
        phases_el = cp.zeros((1, num_modes, indices.shape[0])).astype('f4')

        psi_fl_inner = cp.exp(1j*(r_k1[:,:,cp.newaxis,cp.newaxis].transpose(1,2,3,0)+phases_fl_inner)).sum(-1)
        psi_fl_outer = cp.exp(1j*(r_k2[:,:,cp.newaxis,cp.newaxis].transpose(1,2,3,0)+phases_fl_outer)).sum(-1)
        psi_el = cp.exp(1j*(r_k_el[:,:,cp.newaxis,cp.newaxis].transpose(1,2,3,0)+phases_el)).sum(-1)

        psi_fl_inner *= (pop / pop_max)
        psi_fl_outer *= (pop / pop_max)
        psi_el *= pop / pop_max

        int_fl_inner = cp.abs(psi_fl_inner)**2 
        int_fl_outer = cp.abs(psi_fl_outer)**2 
        int_el = cp.abs(psi_el)**2 

        int2d_fl_inner = int_fl_inner.sum(-1) * kbeta1[cp.newaxis,:] # sum over all modes
        int2d_fl_outer = int_fl_outer.sum(-1) * kbeta2[cp.newaxis,:]
        int2d_el = int_el.sum(-1) * elastic[cp.newaxis,:]

        if self.efilter:
            #int_filter = cundimage.gaussian_filter(int_tot, sigma=(0,self.deltaE), mode='reflect')
            dkernel1 = self.darwin_kernel(cp.max(cp.array([self.dE_b1,2])))
            dkernel2 = self.darwin_kernel(cp.max(cp.array([self.dE_b2,2])))
            int2d_fl_inner = cusignal.fftconvolve(int2d_fl_inner, dkernel1[cp.newaxis,:][:,::-1], mode='same', axes=1)
            int2d_fl_outer = cusignal.fftconvolve(int2d_fl_outer, dkernel2[cp.newaxis,:][:,::-1], mode='same', axes=1)
        
        mode_fl = int2d_fl_inner + int2d_fl_outer
        mode_fl *= self.num_photons / mode_fl.sum()
        int_tot = mode_fl + int2d_el * 0.1 * self.num_photons / int2d_el.sum()
        #int_tot = mode_fl + int2d_el * 10 * self.num_photons / int2d_el.sum()
        int_p = cp.random.poisson(cp.abs(int_tot),size=self.det_shape)
        int_p *= self.adu_phot
        diff_pattern += int_p
        #bg = cp.random.randint(0, diff_pattern.size, self.background)
        #diff_pattern.ravel()[bg] += self.adu_phot

        #gauss_noise = cp.random.normal(self.noise_level,2.5,self.det_shape)
        
        #diff_pattern += gauss_noise
        return diff_pattern

    def calc_beam_profile(self, counter):
        x = cp.arange(-1e4, 1e4, self.mode_period) #(FWHM/num_modes without polarization)
        y = self.num_photons * self.gaussian(x, 1, 0, self.pulse_dur)
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
                        default='sim_beta.ini')
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
    si_dist = float(config.get(section, 'si_dist', fallback=30e-2))
    pixel_size = config.getint(section, 'pixel_size', fallback=100)
    particle_size = config.getfloat(section, 'particle_size', fallback=350)
    line = config.get(section, 'emission_line', fallback='kb1')
    det_shape = fshape
    #num_photons = np.ceil(args.photon_density * det_shape[0] * det_shape[1]).astype(int)
 
    elements = [e for e in config.get(section, 'elements').split()]
    #emission_lines = [l for l in config.get(section, 'emission_lines').split()]
    emission_lines = ['kbeta1,3']
    print('Detector distance: ', det_dist)
    print('Simulate {} line for {}'.format(emission_lines, elements))
 
    sig = Signal(elements, emission_lines, det_shape=det_shape, binning=binning, num_shots=num_shots, num_photons=num_photons, emission_line=line, noise=noise, efilter=efilter, det_dist=det_dist, si_dist=si_dist, pixel_size=pixel_size, particle_size=particle_size)
    sig.create_sample()
    sig.sim_glob()

