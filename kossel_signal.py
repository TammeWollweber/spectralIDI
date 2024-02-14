import numpy as np
from scipy import constants as const

'''
Calculate the number of fluorescent photons which are coherently scattered from the particle itself.
Assume that each atom is emitting 1 photon (see fluence calculation) upstream of the particle.
'''

#Cu
particle_size = 300 #nm
Vnp = 4/3*np.pi*(particle_size/2)**3
atomic_volume = 1.182e-2 #m**3
natoms = Vnp / atomic_volume

sigma_atom = 4.8611e-26 * 1e12 #for 8.047 keV in um**2

fluence = natoms * (1e-6/(particle_size*1e-9))**2

#detector
pixel_size = 75e-6
num_pixels = 1024

#Parameter kalpha
det_dist = 1 #meter
yield_alpha = 0.44109 * (0.57711 + 0.29427)

#nickel_transmission = 0.43

def calc_solid(det_dist, N, pixel_size):
    def calc_omega(l, b, d):
        omega = 4 * np.arcsin(l * b / np.sqrt((l**2+4*d**2) * (b**2+4*d**2)))
        return omega

    omega = calc_omega(pixel_size*N, pixel_size, det_dist)
    return omega


omega_a = calc_solid(det_dist, num_pixels, pixel_size)

Ialpha = fluence * sigma_atom * natoms * omega_a 

print('Particle size: ', particle_size)
print('fluence: ', fluence)
print('natoms: ', natoms)
print('Ialpha: ', Ialpha)


