# -*- coding: utf-8 -*-
"""
Created on Thu May  2 13:44:31 2019

@author: DerMäuser
"""

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
plt.ion()

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
        self.num_modes = num_modes
        self.full_frame = total
        self.alpha_modes = alpha_modes
        self.efilter = efilter

        self.size_emitter = 32
        self.sample = None
        self.center = np.array(self.det_shape)//2 - 1
        self.hits = []
        self.hit_size = None
        self.shots = shots 
        self.loc_scatterer = None
        self.kvector = None
        self.r_k = None
        
        self.adu_phot = 160
        self.fft = None
        self.corr_list = []
        self.noise_level = noise
        
        self.shots_per_file = 1000
        self.file_per_run_counter = 0
        self.data = None
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
        print('Save {} files.'.format(np.ceil(self.shots/self.shots_per_file).astype(int)))

    def create_sample(self):
        self.sample = np.zeros(self.det_shape)
        if self.lines:
            self.sample[30:40, 30:80] = 1
            self.sample[50:60, 30:80] = 1
            self.sample[70:80, 30:80] = 1
        else:
            half_size = int(self.size_emitter/2)
            self.sample[(self.offset-half_size):(self.offset+half_size),(self.offset-half_size):(self.offset+half_size)] = 1
            
        # 1D
        min_val = np.min(np.nonzero(self.sample)[0] - self.center[0])
        max_val = np.max(np.nonzero(self.sample)[0] - self.center[0])
        self.loc_scatterer = np.nonzero(self.sample)[0] - self.center[0]
        #self.loc_scatterer = np.linspace(min_val, max_val, self.num_photons)
        self.kvector = np.arange(self.det_shape[0])
        self.kvector -= self.det_shape[0]//2
        self.kvector = self.kvector * self.kscale_x


    def lorentzian(self, x, x0, a, gam):
            return a * gam**2 / (gam**2 + (x-x0)**2)
       
    def worker(self, i, counter, dict_raw):
        np.random.seed(i+int(time.time()))
        diff_pattern = np.zeros(self.det_shape)
        kx1 = (1024//2+16)//self.binning
        kx2 = (1024//2-17)//self.binning
        kalpha1 = self.lorentzian(np.arange(self.det_shape[1]), kx1, 1, 2.11/self.binning)  
        kalpha2 = self.lorentzian(np.arange(self.det_shape[1]), kx2, 0.5, 2.17/self.binning)  
        num_scatterer = (self.num_photons//3*2, self.num_photons//3)
       
        for m in range(self.num_modes):
            indices = np.arange(self.num_photons)
            np.random.shuffle(indices)
            indices = (indices[:num_scatterer[0]], indices[num_scatterer[0]:])
            spec_mode_conv = np.zeros(self.det_shape)
            for k in range(self.alpha_modes):
                if self.full_frame:
                    if self.incoherent:
                        phases_rand = np.array(np.random.random(size=(num_scatterer[k]//self.num_modes,self.det_shape[1]))*2*np.pi)
                    else:
                        phases_rand = np.zeros((num_scatterer[k],self.det_shape[1]))
                else:
                    if self.incoherent:
                        phases_rand = np.array(np.random.random(size=(num_scatterer[k]//self.num_modes))*2*np.pi)
                    else:
                        phases_rand = np.zeros(num_scatterer[k])
                r_k = np.matmul(self.loc_scatterer[indices[k]%len(self.loc_scatterer),np.newaxis],self.kvector[np.newaxis,:])
                if self.full_frame:
                    psi = np.exp(1j*(r_k[m*(num_scatterer[k]//self.num_modes):(m+1)*(num_scatterer[k]//self.num_modes),:,np.newaxis].transpose(1,0,2)+phases_rand)).sum(1).reshape(self.det_shape)
                else:
                    psi = np.exp(1j*(r_k[m*(num_scatterer[k]//self.num_modes):(m+1)*(num_scatterer[k]//self.num_modes),:].T+phases_rand)).sum(1).reshape(self.det_shape[0])
                mode_int = np.abs(psi)**2
                
                spec_mode = np.zeros(self.det_shape)
                if k == 0:
                    if self.full_frame:
                        spec_mode = mode_int
                    else:
                        spec_mode[:,kx1] = mode_int
                    if self.efilter:
                        mode_conv = signal.fftconvolve(spec_mode, kalpha1[np.newaxis,:], axes=1, mode='same')
                    else:
                        mode_conv = spec_mode
                    spec_mode_conv += mode_conv / mode_conv.sum() * num_scatterer[k]
                else:
                    if self.full_frame:
                        spec_mode = mode_int
                    else:
                        spec_mode[:,kx2] = mode_int
                    if self.efilter:
                        mode_conv = signal.fftconvolve(spec_mode, kalpha2[np.newaxis,:], axes=1, mode='same')
                    else:
                        mode_conv = spec_mode
                    spec_mode_conv += mode_conv / mode_conv.sum() * num_scatterer[k]
                
            #int_filter = ndi.gaussian_filter(spec_mode_conv, sigma=(0,1.13), mode='constant')
            #int_filter = ndi.gaussian_filter(spec_mode_conv, sigma=(0,0), mode='constant')
            int_p = np.random.poisson(np.abs(spec_mode_conv),size=self.det_shape)
            int_p *= self.adu_phot
            diff_pattern += int_p
        
        gauss_noise = np.random.normal(self.noise_level,2.5,self.det_shape)
        diff_pattern += gauss_noise
        dict_raw[i] = diff_pattern

    def simulate(self):
        self.data = []
        self.file_per_run_counter = 0

        if self.shots <= self.shots_per_file:
            file_chunks = 1 
        else:
            file_chunks = np.ceil(self.shots/self.shots_per_file).astype(int)

        ref = np.min((self.shots,self.shots_per_file))
        
        if ref < mp.cpu_count()-1:
            num_cores = ref
            mp_chunks = 1
            mp_step = num_cores
            rest = num_cores
        else:
            num_cores = mp.cpu_count() - 1
            mp_chunks = np.ceil(ref/(num_cores-1)).astype(int)
            mp_step = num_cores - 1
            rest = ref % mp_step

        for l in range(file_chunks):
            self.data = []
            for i in range(mp_chunks):
                manager = mp.Manager()
                dict_raw = manager.dict()
                counter = mp_step*i+l*self.shots_per_file
                if i == (mp_chunks - 1):
                    jobs = [mp.Process(target=self.worker, args=(k,counter,
                                                                 dict_raw)) for
                            k in range(rest)]
                else:
                    jobs = [mp.Process(target=self.worker, args=(k,counter,
                                                                 dict_raw)) for
                            k in range(mp_step)]
                [j.start() for j in jobs]
                [j.join() for j in jobs]
                self.data.extend(np.array(dict_raw.values()))
            
            if self.integrated_signal is None:
                self.integrated_signal = np.zeros_like(self.data[0])
                self.norm = np.zeros_like(self.data[0])
                self.sum_of_corr = np.zeros((self.det_shape[0]*2-1, self.det_shape[1]*2-1))
            self.integrated_signal += np.sum(self.data, 0)

            self.save_raw_data()
            self.file_per_run_counter += 1
        self.run_num += 1

    def save_raw_data(self):
        dpath = self.dir + '{}/'.format(self.exp)
        np.ndarray.tofile((np.array(self.data)/100).astype('u2'), dpath+'Run{}_{:04d}.npy'.format(self.run_num,
                                                self.file_per_run_counter))
       

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
 
    sig = Signal(det_shape=det_shape, binning=args.binning, shots=args.num_shots, num_photons=num_photons, noise=args.noise, num_modes=args.modes, lines=args.lines, incoherent=args.incoherent, efilter=args.filter, total=args.total, alpha_modes=args.alpha)
    sig.create_sample()
    sig.simulate()

