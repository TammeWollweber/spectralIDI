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
import h5py as h5
from utils import decorate_all, info
import argparse
plt.ion()

@decorate_all(info)
class Signal():  
    def __init__(self, det_shape=(135,160), shots=1000, num_photons=50, noise=60, num_modes=1, lines=True, incoherent=False):
        self.det_shape = det_shape
        self.lines = lines
        self.offset = self.det_shape[0]//2 - self.det_shape[0]//4 
        self.kscale_x = 2*np.pi/self.det_shape[0]
        self.kscale_y = 2*np.pi/self.det_shape[1]
        self.num_modes = num_modes

        self.size_emitter = 10
        self.sample = None
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
        self.data = None
        self.run_num = -1
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
            self.sample[(self.offset-half_size):(self.offset+half_size),(self.offset+25-half_size):(self.offset+25+half_size)] = 1
            self.sample[(self.offset+25-half_size):(self.offset+25+half_size),(self.offset+20-half_size):(self.offset+20+half_size)] = 1
            self.sample[(self.offset+25-half_size):(self.offset+25+half_size),(self.offset+40-half_size):(self.offset+40+half_size)] = 1
            
        indices = np.nonzero(self.sample)
        self.num_scatterer = len(indices[0])  # number of scatterers
        self.loc_scatterer = np.zeros((self.num_scatterer,2))
        self.loc_scatterer[:,0] = indices[0][:]
        self.loc_scatterer[:,1] = indices[1][:]
          
        self.kvector = np.zeros((2,) + self.det_shape)
        self.kvector[0,:,:], _ = np.indices(self.det_shape)
        self.kvector[0,:,:] -= self.det_shape[0]//2
        self.kvector[0,:,:] *= self.kscale_x
        self.kvector = self.kvector.reshape(2,self.det_shape[0]*self.det_shape[1])
        
        self.r_k = np.dot(self.loc_scatterer,self.kvector)
        
    def worker(self, i, counter, mean_counts, dict_raw):
        print('\r', counter + i)
        np.random.seed(i+int(time.time()))
        diff_pattern = np.zeros(self.det_shape)
        if not self.incoherent:
            phases_rand = np.zeros((self.num_scatterer,1))
        for m in range(self.num_modes):
            if self.incoherent:
                phases_rand = np.array(np.random.random(size=(self.num_scatterer//self.num_modes,1))*2*np.pi)
            psi = np.exp(1j*(self.r_k[(m*self.num_scatterer//self.num_modes):((m+1)*self.num_scatterer//self.num_modes)]+phases_rand)).sum(0).reshape(self.det_shape)
            mode_intensity = np.multiply(np.conjugate(psi), psi)
            counts = mean_counts//self.num_modes #0.5*my_psi.shape[0]*my_psi.shape[1]        # 0.5 photon counts per pixel, 32768 photons on average per shot
            norm = np.sum(np.sum(mode_intensity,axis=0),axis=0)
            intensity_normalized = np.divide(mode_intensity,norm)*counts
            intensity_poisson = np.random.poisson(np.abs(intensity_normalized),size=intensity_normalized.shape)
            diff_pattern += intensity_poisson

        diff_pattern *= self.adu_phot
        gauss_noise = np.random.normal(self.noise_level,2.5,self.det_shape)
        diff_pattern += gauss_noise

        mask = np.zeros(self.det_shape)
        mask[:,79] = 1
        diff_pattern *= mask

        dict_raw[i] = diff_pattern

    def simulate(self, mean_counts=None):
        if mean_counts is None:
            mean_counts = self.num_photons

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
                                                                 mean_counts, dict_raw)) for
                            k in range(rest)]
                else:
                    jobs = [mp.Process(target=self.worker, args=(k,counter,
                                                                 mean_counts,
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
        np.ndarray.tofile(np.array(self.data).astype('u2'), dpath+'Run{}_{:04d}.npy'.format(self.run_num,
                                                self.file_per_run_counter))
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', nargs='+', type=int, help='det_shape', default=(135,160))
    parser.add_argument('-N', '--num_shots', type=int, default=10)
    parser.add_argument('-M', '--photon_density', type=float, default=0.03, help='Number of photons per pixel')
    parser.add_argument('-n', '--noise', type=int, default=60, help='Noise level')
    parser.add_argument('-m', '--modes', type=int, default=1, help='Number of modes')
    parser.add_argument('-l', '--lines', type=int, default=0, help='Sample shape')
    parser.add_argument('-i', '--incoherent', type=int, default=1, help='Incoherent/coherent simulation')
    args = parser.parse_args()

    print('det_shape: ', args.size)
    print('incoherent: ', args.incoherent)
    det_shape = tuple(args.size)
    #num_photons = np.ceil(args.photon_density * det_shape[0] * det_shape[1]).astype(int)
    num_photons = 10000
 
    sig = Signal(det_shape=det_shape, shots=args.num_shots, num_photons=num_photons, noise=args.noise, num_modes=args.modes, lines=args.lines, incoherent=args.incoherent)
    sig.create_sample()
    sig.simulate()

