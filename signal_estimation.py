import numpy as np
from scipy import constants as const

#Mn (alpha crystal structure)
Vunit = 320.085/1000 #nm**3
atoms_unit = 58
particle_size = 100 #nm
Vnp = 4/3*np.pi*(particle_size/2)**3
natoms = Vnp / Vunit * atoms_unit
mu = 3.063e-6

#Xray pulse
I0 = 100e-6 #Joule
Iev = I0 / const.e
E = 6.58e3 #photon energy
num_photons = Iev / E
focus_size = 300 #nm
fluence = (particle_size/focus_size)**2 * num_photons

#detector
pixel_size = 75e-6
num_pixels = 1024
si_dist = 30e-3 #meter

#Parameter kalpha
det_dist_a = 1 #meter
darwin_width_a = 9.27 #arcsec
yield_alpha = 0.31018 * (0.58134 + 0.29394)

#Parameter kbeta
det_dist_b = 2.5 #meter
darwin_width_b = 5.176 #arcsec
yield_beta = 0.31018 * 0.08153

def calc_Pabs(mu, particle_size, unit_size, Nunit):
    Iabs = 1 - np.exp(-particle_size*1e-9/mu)
    N_abs = particle_size**3/unit_size * Nunit
    Patom = Iabs / N_abs
    return Patom

def calc_solid(det_dist, si_dist, darwin_width):
    def calc_omega(l, b, d):
        omega = 4 * np.arcsin(l * b / np.sqrt((l**2+4*d**2) * (b**2+4*d**2)))
        return omega

    def calc_det_eff(d1, d2, pixel_size=75e-6, num_pixels=1024):
        h = pixel_size*num_pixels
        return h/d1*d2

    def calc_width_eff(theta, d=30e-3):
        return 2 * d * np.tan(theta/3600/2)

    b = calc_det_eff(det_dist, si_dist)
    l = calc_width_eff(darwin_width, si_dist)
    omega = calc_omega(l, b, si_dist)
    return omega


patom = calc_Pabs(mu, particle_size, Vunit, atoms_unit)

omega_a = calc_solid(det_dist_a, si_dist, darwin_width_a)
omega_b = calc_solid(det_dist_b, si_dist, darwin_width_b)

Ialpha = fluence * patom * natoms * yield_alpha * omega_a
Ibeta = fluence * patom * natoms * yield_beta * omega_b


