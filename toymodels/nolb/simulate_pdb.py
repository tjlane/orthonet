
import os, sys
import numpy as np
import argparse

from dmach.temsim import simutils
from dmach.temsim import cryoemio

home = os.environ['HOME']
SIMULATOR_BIN= home + '/bin/TEM-simulator'
print('USING:', SIMULATOR_BIN)


# =============================================================================

parser = argparse.ArgumentParser(description='simulate 48 particles of given PDB')
parser.add_argument('--pdb', help='PDB file to simulate')
parser.add_argument('--outdir', help='output director')
parser.add_argument('--keyword', default='sim', help='name for the specific simulation')
args = parser.parse_args()

# =============================================================================

PDB        = args.pdb
OUTPUT_DIR = args.outdir
KEYWORD    = args.keyword
SEED       = np.random.randint(999999)

# =============================================================================
# ---------------                PARAMETERS          --------------------------

# === molecular model
voxel_size      = 0.1                # [nm]
particle_name   = 'ribosome'         # Name of the particle. Not very important.
particle_mrcout = None               # volume map of sample is written.

# === specimen grid
hole_diameter         = 1200 # [nm]
hole_thickness_center = 100  # [nm]
hole_thickness_edge   = 100  # [nm]

# === beam
voltage           = 300 # [kV]
energy_spread     = 1.3 # [V]
electron_dose     = 100 # [e/nm**2] dose per image
electron_dose_std = 0   # standard deviation of dose per image

# === optics
magnification         = 81000 #
spherical_aberration  = 2.7   # [mm]
chromatic_aberration  = 2.7   # [mm]
aperture_diameter     = 50    # [um] in back focal plane
focal_length          = 3.5   # [mm]
aperture_angle        = 0.1   # [mrad] of the condenser lens
defocus               = 1.0   # [um]
defocus_syst_error    = 0.0   #
defocus_nonsyst_error = 0.0   #
optics_defocusout     = None  # file to write defocus values

# === detector
detector_Nx           = 5760 # number of pixels along X axis
detector_Ny           = 4092 # number of pixels along Y axis
detector_pixel_size   = 5    # [um]
detector_gain         = 32   # average number of counts per electron
detector_Q_efficiency = 0.5  # detector quantum efficiency
noise                 = 'no' # whether quantized electron waves result in noise
MTF_params            = [0,0,1,0,0] # to be described. [0,0,1,0,0] is perfect


# =============================================================================

sample_dimensions = [ hole_diameter, hole_thickness_center, hole_thickness_edge ]
beam_params       = [ voltage, energy_spread, electron_dose, electron_dose_std ]
optics_params     = [ magnification, spherical_aberration, chromatic_aberration, 
                      aperture_diameter, focal_length, aperture_angle,
                      defocus, defocus_syst_error, defocus_nonsyst_error, optics_defocusout ]
detector_params   = [ detector_Nx, detector_Ny, detector_pixel_size,
                      detector_gain, noise, detector_Q_efficiency,
                      MTF_params[0], MTF_params[1], MTF_params[2], MTF_params[3], MTF_params[4] ]

# =============================================================================


# Outputs
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

pdb_file, mrc_file, crd_file, log_file, inp_file, h5_file = cryoemio.simio(PDB, OUTPUT_DIR, KEYWORD)
if particle_mrcout is not None:
    particle_mrcout = mrc_file


x_range, y_range, numpart = simutils.define_grid_in_fov(sample_dimensions, 
                                                        optics_params, 
                                                        detector_params, 
                                                        pdb_file=pdb_file, 
                                                        Dmax=30, pad=5.)

simutils.write_crd_file(numpart, xrange=x_range, yrange=y_range, crd_file=crd_file)

params_dictionary = simutils.fill_parameters_dictionary(mrc_file = mrc_file, 
                                                        pdb_file = pdb_file, 
                                                        particle_mrcout = particle_mrcout, 
                                                        crd_file = crd_file,
                                                        sample_dimensions = sample_dimensions,
                                                        beam_params = beam_params,
                                                        optics_params = optics_params,
                                                        detector_params = detector_params,
                                                        log_file = log_file,
                                                        seed=SEED)

simutils.write_inp_file(inp_file=inp_file, dict_params=params_dictionary)

cmd = '{0} {1}'.format(SIMULATOR_BIN, inp_file)
print('EXEC:', cmd)
ret = os.system(cmd)
if ret != 0:
    raise RuntimeError('TEM-simulator returned non-zero exit: %d' % ret)


# particle picking
data = cryoemio.mrc2data(mrc_file = mrc_file)
micrograph = data[0,...]
particles = simutils.microgaph2particles(micrograph, sample_dimensions, 
                                         optics_params, detector_params, 
                                         pdb_file=pdb_file, Dmax=30, pad=5.)

# save
cryoemio.data_and_dic_2hdf5(particles, h5_file, dic = params_dictionary)
cryoemio.add_crd_to_h5(input_h5file=h5_file, input_crdfile=crd_file, output_h5file=h5_file)

print('finished.')



