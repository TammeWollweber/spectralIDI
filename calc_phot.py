'''
solid angle and number of photons calculation for xray-spectral IDI simulation. Does 1000 photons/shot make sense?
'''

import numpy as np


#Fabians constants
pdens_fabian = 7.7e-3
det_dist_fabian = 8
pix_size_fabian = 200e-6
pvol_fabian = 2*np.pi*(200e-9)**2 * 20e-6 #effective particle size in Fabian experiment: 2 dots with 250um radius on 20um Cu foil, assuming 100% excitation
dot_size = 400e-9
dot_sep = 1e-6
speckle_size_fabian = 48 #in pixel

#Detector constants
det_dist = 1 # in meter
det_shape = (1024,1024) # in pix
pix_size = 100e-6

#Cu nanoparticle constants
atoms_unit = 4 #fcc crystal structure has 4 atoms per unit cell
unit_cell_param = 3.61e-10 #unit cell parameter
psize = 200e-9 #100nm NP
pvol= 4/3*np.pi*(psize)**3 #size of a 100nm sphere
num_atoms = pvol / unit_cell_param**3 * atoms_unit
k_eff = 0.44 #fluorescence yield of all k lines
kb5_eff = 7.9e-4 #relative intensity of k_eff
kb13_eff = 0.12 #relative intensity of k_eff

#Silicon crystal constants
si_dist = 30e-3 # in meter
si_size = (110e-3, 25e-3) # in m
si_eff = 0.01 #reflection efficiency of (440) reflection (guessed)

def solid_angle(l, b, d):
    return 4 * np.arcsin((l*b)/np.sqrt((l**2+4*d**2)*(b**2+4*d**2)))

s_fabian = solid_angle(det_shape[0]*pix_size, det_shape[1]*pix_size, 8)
s_det = solid_angle(det_shape[0]*pix_size, det_shape[1]*pix_size, det_dist)
s_si = solid_angle(si_size[0], si_size[1], si_dist)

'''
calculate effective det size at silicon distance
tan(phi) = det_shape[0] * pix_size / det_dist
tan(phi) = x / si_dist
'''

x = si_dist * det_shape[0] * pix_size / det_dist
s_eff = solid_angle(si_size[0], x, si_dist)

num_phot_fabian = pdens_fabian * np.prod(det_shape)

num_phot_kb5_solid = si_eff * kb5_eff * pvol/pvol_fabian * (pix_size/pix_size_fabian)**2 *(s_eff/s_fabian)**2 * num_phot_fabian
num_phot_kb5 = si_eff * kb5_eff * k_eff * s_eff * num_atoms
num_phot_kb13 = si_eff * kb13_eff * k_eff * s_eff * num_atoms


'''
Calculate speckle size and energy resolution for giving parameters
'''
speckle_size = dot_size / psize * pix_size_fabian/pix_size * det_dist/det_dist_fabian * speckle_size_fabian 

e1 = 8975 #kbeta_2,5 Cu in eV
e2 = 8985 #kbeta_2,5 Cu1+
e3 = 9000 #elastic 
phi1 = 46.01
phi2 = 45.95
phi3 = 45.85
e_sep = e3-e1
e_center = np.round((e1+e3)/2).astype(int)
pix_sep = det_dist * np.tan((phi1-phi3)*np.pi/180) / pix_size
e_res = (e_sep)/np.round(pix_sep)

print('Estimated number of photons per shot from solid angle calculus: ', num_phot_kb5_solid)
print('Estimated number of photons per shot for kb5: ', num_phot_kb5)
print('Estimated number of photons per shot for kb13: ', num_phot_kb13)

print('Estimated central speckle size: ', speckle_size)
print('Estimated energy resolution: ', e_res)
