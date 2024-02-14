import numpy as np
from scipy import constants as const

#Cu
particle_size = 300 #nm
Vnp = 4/3*np.pi*(particle_size/2)**3
atomic_volume = 1.182e-2 #m**3
natoms = Vnp / atomic_volume

sigma = 274.9*1e-4 #m**2/g
atomic_weight = 63.546
amu = 1.6605e-24 #g
mass_atom = amu * atomic_weight # g
sigma_atom = sigma * mass_atom * 1e12 #um**2

#Xray pulse
I0 = 300e-6 #Joule pulse energy in 3um**2 focus
Iev = I0 / const.e
E = 9.1e3 #photon energy
num_photons = Iev / E # per pulse

focus_size = 3 #um
fluence = num_photons * (1e-6/(focus_size*1e-6))**2

#detector
pixel_size = 75e-6
num_pixels = 1024
si_dist = 30e-2 #meter

#Parameter kalpha
det_dist_a = 1 #meter
yield_alpha = 0.44109 * (0.57711 + 0.29427)

#Parameter kbeta
det_dist_b = 1 #meter
yield_beta = 0.44109 * (0.08401 + 0.04352)


def normed_lorentzian(x, x0, a, gam):
    return 1/np.pi * 1/2*gam / ((1/2*gam)**2 + (x-x0)**2)

def calc_solid(det_dist, N, pixel_size):
    def calc_omega(l, b, d):
        omega = 4 * np.arcsin(l * b / np.sqrt((l**2+4*d**2) * (b**2+4*d**2)))
        return omega

    omega = calc_omega(pixel_size*N, pixel_size, det_dist)
    return omega


omega_a = calc_solid(det_dist_a, num_pixels, pixel_size)
omega_b = calc_solid(det_dist_b, num_pixels, pixel_size)

spec_a = normed_lorentzian(np.arange(1024)-512, 0, 1, 2.17/0.2)
spec_b = normed_lorentzian(np.arange(1024)-512, 0, 1, 3.7/0.28)

Ialpha = np.min([fluence * sigma_atom, 1]) * natoms * yield_alpha * (omega_a*spec_a).sum()
Ibeta =  np.min([fluence * sigma_atom, 1]) * natoms * yield_beta * (omega_b*spec_b).sum()

print('Particle size: ', particle_size)
print('fluence: ', fluence)
print('natoms: ', natoms)
print('Ialpha: ', Ialpha)
print('Ibeta: ', Ibeta)


