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
from scipy import constants as const
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
    def __init__(self, elements, emission_lines, det_shape=(1024,1024), binning=8, num_shots=1000, num_photons=50,
                 noise=60, incoherent=False, efilter=False, alpha_modes=2, det_dist=4, si_dist=30e-2, pixel_size=100, particle_size=350):

        self.elements = elements
        self.lines = emission_lines
        self.specs = []

        self.det_shape = tuple(np.array(det_shape) // binning)
        print('det_shape: ', self.det_shape)        
        print('alpha mode: ', alpha_modes)
        self.binning = binning
        self.offset = self.det_shape[0]//2
        self.si_dist = si_dist
        self.det_distance = det_dist
        self.det_dist_E = det_dist-si_dist
        self.pixel_size = pixel_size*1e-6*self.binning
        self.efilter = efilter
        self.alpha_modes = alpha_modes
        self.num_photons = num_photons

        self.sample_shape = (256, 256)
        self.sample = None
        self.hits = []
        self.hit_size = None
        self.num_shots = num_shots 
        self.num_scatterer = None
        self.kvector = None
        self.r_k = None
        self.beat_period = 413 #attoseconds
        self.mode_period = None
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
        self.run_num = 0
        self.exp = None
        #self.dir = '/mpsd/cni/processed/wittetam/spectral_sim/raw/'
        self.dir = '/scratch/wittetam/spectral_sim/raw/'
        self.num_cores = None
        self.integrated_signal = None
        self.incoherent = incoherent

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
                self.specs.append(config[e][l])

        self.specs.append(config[e]['elastic'])
        self.tau = int(config[e]['tau'])
        self.width = float(config[e]['kalpha2']['w'])
        print('coherence time, width kalpha2: ', self.tau, self.width)
        print(self.specs)

    def _calc_energy_resolution(self, counter):
        lam_cen = const.h * const.c / (self.E* const.e)
        tcen = 71.62
        lat = lam_cen / (2*np.sin(tcen/180*np.pi)) #lattice constant
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
        darwin_a1 = float(self.specs[0]['darwin'])
        darwin_a2 = float(self.specs[1]['darwin'])
        e_sep = E1 - E2
        tcen = (phi1+phi2)/2

        self.E = np.round((E1 + E2)/2).astype(int)
        self._calc_energy_resolution(counter)


        self.kvector = (2*cp.sin(0.5*cp.arctan((cp.arange(self.det_shape[1])-self.det_shape[1]//2)*self.pixel_size/self.det_distance))/(1239.84/self.E)).astype('f4')

        qmax = cp.max(np.abs(self.kvector))
        self.rscale = 1/(2*qmax)
        self.size_em = cp.rint((self.particle_size / self.rscale - 1) / 2).get().astype(int)
 
        x1 = self.det_dist_E * np.tan(np.abs(phi1-tcen)*cp.pi/180)
        x2 = self.det_dist_E * np.tan(np.abs(phi2-tcen)*cp.pi/180)
        self.pix_sep = np.abs(x1+x2) / self.pixel_size

        self.xka2 = self.det_shape[1]//2 - self.pix_sep//2
        self.xka1 = self.det_shape[1]//2 + self.pix_sep//2

        self.dE_a1 = np.ceil(2*self.det_dist_E * (np.tan((phi1+darwin_a1/3600-tcen)*np.pi/180) - np.tan((phi1-tcen)*np.pi/180)) / self.pixel_size).astype(int) #uncertainty from darwin plateau in units of pixel
        self.dE_a2 = np.ceil(2*self.det_dist_E * (np.tan((phi2+darwin_a2/3600-tcen)*np.pi/180) - np.tan((phi2-tcen)*np.pi/180)) / self.pixel_size).astype(int) #uncertainty from darwin plateau in units of pixel
        self.deltaE = np.max((self.dE_a1, self.dE_a2))

        self.mode_period = self.tau * self.width / (self.deltaE * self.e_res)

        self.create_sample()
        self.sample = cp.array(self.sample)
        self.psample = cp.array(self.psample)
        if counter == 0:
            print('phi: ', phi1, phi2, tcen)
            print('rscale: ', self.rscale)
            print('size_em: ', self.size_em)
            print('cen, pix_sep: ', self.E, self.pix_sep)
            print('Energy resolution from pixels: ', self.e_res)
            print('Uncertainty from darwin: ', self.dE_a1, self.dE_a2)
            print('dE [eV], [pix]: ', self.deltaE*self.e_res, self.deltaE)
            print('mode_period: ', self.mode_period)
 
    def create_sample(self):
        mask = np.zeros(self.sample_shape)
        self.sample = np.zeros(self.sample_shape)
        center = np.array(self.sample.shape)//2
        X,Y = np.meshgrid(np.arange(self.sample_shape[0])-center[0], np.arange(self.sample_shape[1])-center[1], indexing='ij')
        r = np.sqrt(X**2 + Y**2)
        mask[r<self.size_em] = 1
        self.sample = 2*np.sqrt(np.abs(self.size_em**2-r**2)*mask)
        self.psample = self.sample.sum(-1)
            
    def lorentzian(self, x, x0, a, gam):
        return a * gam**2 / (gam**2 + (x-x0)**2)

    def gaussian(self, x, a, mu, sig):
        sig = sig/2.355
        return a/(sig*cp.sqrt(2*cp.pi)) * cp.exp(-(x-mu)**2/(2*sig**2))
    
    def darwin_kernel(self, width):
        x0 = int(self.det_shape[1])//2
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
            sys.stderr.write('%d, %.3f s/file\n' % (self.file_counter, (time.time()-stime)/(i+1)))
            

    def sim_file(self, counter):
        num_jobs = NUM_DEV * JOBS_PER_DEV
        self._sim_frame(counter)
        if counter % num_jobs == 0:
            sys.stderr.write('\r%s'%(counter))
    
    def _sim_frame(self, counter):
        kalpha1 = self.lorentzian(cp.arange(self.det_shape[1]), self.xka1, 1, float(self.specs[0]['w'])/self.e_res)  
        kalpha2 = self.lorentzian(cp.arange(self.det_shape[1]), self.xka2, 0.5, float(self.specs[1]['w'])/self.e_res)  
        kspec = cp.sqrt(cp.array([kalpha1, kalpha2]))
        spectrum = cp.sqrt(kalpha1 + kalpha2)
        
        pop = self.calc_beam_profile(counter)
        pop_max = cp.round(pop.max()).astype(int)
        num_modes = len(pop)
        if counter == 0:
            print('num modes: ', num_modes)

        size = len(cp.where(self.sample != 0)[0])
        indices = cp.random.choice(cp.arange(0,self.sample.shape[0]), size=size, p=self.psample/self.psample.sum())
        num_chunks = np.max((16//self.binning**2,1))
        chunk_size = indices.shape[0]//num_chunks

        intens = cp.zeros(self.det_shape)
        psi_spec = None
        for i in range(num_chunks):
            r_k = cp.outer(indices[i*chunk_size:(i+1)*chunk_size]*self.rscale, self.kvector)
            #r_k = cp.outer(indices*self.rscale,self.kvector)
            if self.incoherent:
                phases_rand = cp.array(cp.random.random(size=(2, num_modes, chunk_size)))
            else:
                phases_rand = cp.zeros((2, num_modes, chunk_size))

            psi = cp.exp(1j*2*cp.pi*(r_k[:,:,cp.newaxis,cp.newaxis].transpose(1,2,3,0)+phases_rand)).sum(-1)
            if self.alpha_modes == 1:
                if i == 0:
                    psi_spec = (psi.mean(1).transpose(1,0)[:,:,cp.newaxis] * spectrum[cp.newaxis,:])
                else:
                    psi_spec += (psi.mean(1).transpose(1,0)[:,:,cp.newaxis] * spectrum[cp.newaxis,:])

            elif self.alpha_modes == 2:
                if i == 0:
                    psi_spec = (psi.T[:,:,:,cp.newaxis] * kspec[cp.newaxis,cp.newaxis,:,:].transpose(0,2,1,3)).transpose(0,2,3,1)
                else:
                    psi_spec += (psi.T[:,:,:,cp.newaxis] * kspec[cp.newaxis,cp.newaxis,:,:].transpose(0,2,1,3)).transpose(0,2,3,1)

            elif self.alpha_modes == 3:
                if i == 0:
                    psi_spec_i = (psi.T[:,:,:,cp.newaxis] * kspec[cp.newaxis,cp.newaxis,:,:].transpose(0,2,1,3)).transpose(0,2,3,1)
                
                    beat_phases = (cp.arange(num_modes) * self.mode_period/self.beat_period * 2*cp.pi) % (2*cp.pi)
                    psi_spec = 1/2 * (psi_spec[:,:,:,0] + (psi_spec_i[:,:,:,1].T * cp.exp(1j*beat_phases)).T)

        psi_spec = (psi_spec.T *  (pop / pop_max)).T
        if self.alpha_modes == 1:
            mode_int = cp.abs(psi_spec)**2
            int_tot = mode_int.sum(0)
        elif self.alpha_modes == 2:
            mode_int = cp.abs(psi_spec)**2
            int_tot = mode_int.sum(0).mean(-1)
        elif self.alpha_modes == 3:
            mode_int = cp.abs(psi_spec)**2
            int_tot = mode_int.sum(0)

        int_tot /= int_tot.sum() / self.num_photons
        if self.efilter:
            #int_filter = cundimage.gaussian_filter(int_tot, sigma=(0,self.deltaE), mode='reflect')
            dkernel1 = self.darwin_kernel(cp.max(cp.array([self.dE_a1,2])))
            dkernel2 = self.darwin_kernel(cp.max(cp.array([self.dE_a2,2])))
            int_filter = cusignal.fftconvolve(int_tot, dkernel1[cp.newaxis,:][:,::-1], mode='same', axes=1)
            int_filter = cusignal.fftconvolve(int_tot, dkernel2[cp.newaxis,:][:,::-1], mode='same', axes=1)
        else:
            int_filter = int_tot
        int_p = cp.random.poisson(cp.abs(int_filter),size=self.det_shape)
        int_p *= self.adu_phot
        self.data += int_p
        #bg = cp.random.randint(0, diff_pattern.size, self.background)
        #diff_pattern.ravel()[bg] += self.adu_phot

        #gauss_noise = cp.random.normal(self.noise_level,2.5,self.det_shape)
        
        #diff_pattern += gauss_noise

    def calc_beam_profile(self, counter):
        x = cp.arange(-1e4, 1e4, self.mode_period)
        y = self.num_photons * self.gaussian(x, self.mode_period, 0, self.pulse_dur)
        shot_noise = cp.random.uniform(-0.8, 0.8, len(y))
        y_noise = cp.round(y + shot_noise * y).astype(int)
        mask = cp.zeros_like(y_noise)
        mask[y_noise>=4] = 1 #at least 4 photons per mode, so 2 for each polarization
        y_final = (y_noise * mask)//2
        #cp.save('beam_profile_{}.npy'.format(counter), y_noise)
        y_final = y_final[y_final!=0]
        return cp.repeat(y_final,2) / 2


    def save_file(self):
        dpath = self.dir + '{}/'.format(self.exp)
        np.ndarray.tofile(self.data.astype('u2'), dpath+'Run{}_{:04d}.npy'.format(self.run_num, self.file_counter))
        self.file_counter += 1
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate 1 spectral-IDI')
    parser.add_argument('-c', '--config_fname', help='Config file',
                        default='sim_alpha.ini')
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
    incoherent = config.getboolean(section, 'incoherent', fallback=True)
    efilter = config.getboolean(section, 'filter', fallback=True)
    alpha = config.getint(section, 'alpha', fallback=2)
    det_dist = config.getfloat(section, 'det_dist', fallback=1.)
    si_dist = config.getfloat(section, 'si_dist', fallback=30e-2)
    pixel_size = config.getint(section, 'pixel_size', fallback=100)
    particle_size = config.getint(section, 'particle_size', fallback=350)
    det_shape = fshape
    #num_photons = np.ceil(args.photon_density * det_shape[0] * det_shape[1]).astype(int)

    elements = [e for e in config.get(section, 'elements').split()]
    #emission_lines = [l for l in config.get(section, 'emission_lines').split()]
    emission_lines = ['kalpha1', 'kalpha2']
    file_chunk = 1000
    num_files = np.ceil(num_shots/file_chunk).astype(int)

    print('Simulate {} line for {}'.format(emission_lines, elements))
 
    sig = Signal(elements, emission_lines, det_shape=det_shape, binning=binning, num_shots=num_shots, num_photons=num_photons, 
                 noise=noise, incoherent=incoherent, efilter=efilter, alpha_modes=alpha, det_dist=det_dist, si_dist=si_dist, pixel_size=pixel_size, particle_size=particle_size)

    for i in range(num_files):
        sig.sim_glob()
        sig.save_file()

