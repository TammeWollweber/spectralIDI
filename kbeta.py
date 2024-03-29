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

NUM_DEV = 2
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

        self.size_em1 = None
        self.size_em2 = None
        self.sample_shape = (256, 256)
        self.sample = None
        self.hits = []
        self.hit_size = None
        self.num_shots = num_shots 
        self.kvector = None
        self.r_k = None
        self.adu_phot = 1
        self.fft = None
        self.corr_list = []
        self.noise_level = noise
        self.background = np.round(10*self.num_photons).astype(int) #percentage of num_photons
        if self.det_shape[0] <= 1024:
            self.shots_per_file = 1000
        else:
            self.shots_per_file = 250
        if self.num_shots < self.shots_per_file:
            self.shots_per_file = self.num_shots
        self.file_counter = 0
 
        self.data = []
        self.run_num = 0
        self.exp = None
        self.dir = '/scratch/wittetam/spectral_sim/raw/'
        #self.dir = '/mpsd/cni/processed/wittetam/spectral_sim/raw/'
        self.num_cores = None
        self.integrated_signal = None
        self.tau = None
        self.width = None
        self.pulse_dur = 6200
        self.particle_size = particle_size
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

    def _calc_energy_resolution(self,  counter):
        lam_cen = const.h * const.c / (self.E * const.e)
        tcen = 58.84 
        lat = lam_cen / (2*np.sin(tcen/180*np.pi))
        phi = np.arctan(1024*self.pixel_size/self.det_distance)*180/np.pi
        tmax = tcen + phi
        tmin = tcen - phi
        lam_max = 2*lat*np.sin(tmax*np.pi/180)
        lam_min = 2*lat*np.sin(tmin*np.pi/180)
        Emax = const.h * const.c / lam_max / const.e
        Emin = const.h * const.c / lam_min / const.e
        self.e_res = np.abs(Emax-Emin) / 1024
        if counter == 0:
            print('Energy range: ', Emax, Emin, np.abs(Emax-Emin))
            print('Energy resolution: ', self.e_res)



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
        self.E = np.round((E1+E3)/2).astype(int)
        self._calc_energy_resolution(counter)

        phi_cen = (phi1+phi3)/2
        self.kvector = (2*cp.sin(0.5*cp.arctan((cp.arange(self.det_shape[1])-self.det_shape[1]//2)*self.pixel_size/self.det_distance))/(1239.84/self.E)).astype('f4')

        qmax = cp.max(np.abs(self.kvector))
        self.rscale = 1/(2*qmax)
        self.size_em1 = cp.rint((self.particle_size / self.rscale - 1) / 2).get().astype(int)
        self.size_em2 = np.ceil(self.size_em1/0.7).astype(int)
 
        print('size emitter: ', self.size_em1, self.size_em2)
        x1 = self.det_dist_E * np.tan(np.abs(phi1-phi_cen)*cp.pi/180)
        x2 = self.det_dist_E * np.tan(np.abs(phi2-phi_cen)*cp.pi/180)
        x3 = self.det_dist_E * np.tan(np.abs(phi3-phi_cen)*cp.pi/180)
        self.pix_sep = np.abs(x1+x3) / self.pixel_size
        self.beta_shift =  np.abs(x1-x2) / self.pixel_size
 
        self.dE_b1 = np.ceil(2*self.det_dist_E * (np.tan((phi1+darwin_b1/3600-phi_cen)*np.pi/180) - np.tan((phi1-phi_cen)*np.pi/180)) / self.pixel_size).astype(int) #uncertainty from darwin plateau in units of pixel
        self.dE_b2 = np.ceil(2*self.det_dist_E * (np.tan((phi2+darwin_b2/3600-phi_cen)*np.pi/180) - np.tan((phi2-phi_cen)*np.pi/180)) / self.pixel_size).astype(int) #uncertainty from darwin plateau in units of pixel
        self.deltaE = np.max((self.dE_b1, self.dE_b2))

        self.mode_period = self.tau * self.width / (self.deltaE * self.e_res)
 
        self.xkb1 = self.det_shape[1]//2 - self.pix_sep//2
        self.xkb2 = self.det_shape[1]//2 - self.pix_sep//2 + self.beta_shift
        self.xkel = self.det_shape[1]//2 + self.pix_sep//2

       
        self.create_sample()
        self.sample = cp.array(self.sample)
    
        if counter == 0:
            print('phi: ', phi1, phi2, phi3, phi_cen)
            print('pix_sep: ', self.pix_sep)
            print('bshift: ', self.beta_shift)
            print('Energy resolution from pixels: ', self.e_res)
            print('Uncertainty from darwin: ', self.dE_b1, self.dE_b2)
            print('ecenter: ', self.E)
            print('mode_period: ', self.mode_period)
            print('dE [eV], [pix]: ', self.deltaE*self.e_res, self.deltaE)

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
        num_jobs = NUM_DEV * JOBS_PER_DEV
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass

        data_arr = mp.Array(ctypes.c_double, self.shots_per_file*int(np.product(self.det_shape)))
        jobs = [mp.Process(target=self.worker, args=(d, data_arr)) for
                    d in range(num_jobs)]
        [j.start() for j in jobs]
        [j.join() for j in jobs]

        self.data = np.frombuffer(data_arr.get_obj()).reshape((self.shots_per_file, ) + self.det_shape)
      
    def worker(self, rank, data):
        num_jobs = NUM_DEV * JOBS_PER_DEV
        devnum = rank // JOBS_PER_DEV
        cp.cuda.Device(devnum).use()
        stime = time.time()

        cp.random.seed((os.getpid() * int(time.time())) % 123456789)
        mydata = np.frombuffer(data.get_obj()).reshape((self.shots_per_file,) + self.det_shape)
        self._init_sim(rank)

        for i, _ in enumerate(np.arange(self.shots_per_file)[rank::num_jobs]):
            self.data = cp.zeros(self.det_shape)
            idx = i*num_jobs+rank
            self.sim_file(idx)
            mydata[idx] = self.data.get()
        if rank == 0:
            sys.stderr.write('%d,  %.3f s/file\n' % (self.file_counter, (time.time()-stime) / (i+1)))

    def sim_file(self, counter):
        num_jobs = NUM_DEV * JOBS_PER_DEV
        self._sim_frame(counter)
        if counter % num_jobs == 0:
            sys.stderr.write('\r%s'%(counter))
        
    def _sim_frame(self, counter):
        kbeta1 = self.lorentzian(cp.arange(self.det_shape[1]), 1, self.xkb1, 2.97/self.e_res)  
        kbeta2 = self.lorentzian(cp.arange(self.det_shape[1]), 1, self.xkb2, 2.97/self.e_res)  
        elastic = cp.sqrt(self.gaussian(cp.arange(self.det_shape[1]), 0.1 ,self.xkel, 9/self.e_res))
        spectrum = kbeta1 + kbeta2
        
        pop = self.calc_beam_profile(counter)
        pop_max = cp.round(pop.max()).astype(int)
        num_modes = len(pop)

        inner_weight = self.p_inner.sum()
        outer_weight = self.p_outer.sum()

        size = len(cp.where(self.sample != 0)[0])

        indices = cp.random.choice(cp.arange(0,self.sample.shape[0]), size=size, p=self.p_tot/self.p_tot.sum()).astype('u2')
        ind_inner = cp.random.choice(cp.arange(0,self.sample.shape[0]), size=np.round(size*(inner_weight/(inner_weight+outer_weight))).astype(int), p=self.p_inner/self.p_inner.sum()).astype('u2')
        ind_outer = cp.random.choice(cp.arange(0,self.sample.shape[0]), size=np.round(size*(outer_weight/(inner_weight+outer_weight))).astype(int), p=self.p_outer/self.p_outer.sum()).astype('u2')

        if counter == 0:
            print('num modes: ', num_modes)

        r_k_el = cp.outer(indices*self.rscale,self.kvector)
        r_k1 = cp.outer(ind_inner*self.rscale,self.kvector)
        r_k2 = cp.outer(ind_outer*self.rscale,self.kvector)
        phases_fl_inner = cp.array(cp.random.random(size=(num_modes, ind_inner.shape[0]))).astype('f4')

        phases_fl_outer = cp.array(cp.random.random(size=(num_modes, ind_outer.shape[0]))).astype('f4')
        phases_el = cp.zeros((1, num_modes, indices.shape[0])).astype('f4')

        psi_fl_inner = cp.exp(1j*2*cp.pi*(r_k1[:,:,cp.newaxis,cp.newaxis].transpose(1,2,3,0)+phases_fl_inner)).sum(-1)
        psi_fl_outer = cp.exp(1j*2*cp.pi*(r_k2[:,:,cp.newaxis,cp.newaxis].transpose(1,2,3,0)+phases_fl_outer)).sum(-1)
        psi_el = cp.exp(1j*2*cp.pi*(r_k_el[:,:,cp.newaxis,cp.newaxis].transpose(1,2,3,0)+phases_el)).sum(-1)

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
        #int_tot = mode_fl + int2d_el * 0.1 * self.num_photons / int2d_el.sum()
        int_tot = mode_fl + int2d_el * 10 * self.num_photons / int2d_el.sum()
        int_p = cp.random.poisson(cp.abs(int_tot),size=self.det_shape)
        int_p *= self.adu_phot
        self.data += int_p
        #bg = cp.random.randint(0, diff_pattern.size, self.background)
        #diff_pattern.ravel()[bg] += self.adu_phot

        #gauss_noise = cp.random.normal(self.noise_level,2.5,self.det_shape)
        
        #diff_pattern += gauss_noise

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


    def save_file(self):
        dpath = self.dir + '{}/'.format(self.exp)
        np.ndarray.tofile(self.data.astype('u2'), dpath+'Run{}_{:04d}.npy'.format(self.run_num,self.file_counter))
        self.file_counter += 1
       

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
    file_chunk = 1000
    num_files = np.ceil(num_shots/file_chunk).astype(int)
    print('num files: ', num_files)
 
    sig = Signal(elements, emission_lines, det_shape=det_shape, binning=binning, num_shots=num_shots, num_photons=num_photons, emission_line=line, noise=noise, efilter=efilter, det_dist=det_dist, si_dist=si_dist, pixel_size=pixel_size, particle_size=particle_size)
    
    for i in range(num_files):
        sig.sim_glob()
        sig.save_file()

