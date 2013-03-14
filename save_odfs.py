import nibabel as nib
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.odf import gfa
from dipy.reconst.dsi import DiffusionSpectrumDeconvModel
from dipy.data import get_sphere
from dipy.viz.mayavi.spheres import show_odfs
from load_data import get_train_dsi


data, affine, gtab = get_train_dsi(30)

gqi_model = GeneralizedQSamplingModel(gtab,
                                      method='gqi2',
                                      sampling_length=3,
                                      normalize_peaks=True)

crop = 20
gqi_fit = gqi_model.fit(data[crop:29,crop:29,crop:29])
sphere = get_sphere('symmetric724')
gqi_odf = gqi_fit.odf(sphere)
gqi_gfa = gfa(gqi_odf)

import nibabel as nib

affine[:3,3] +=  crop 
print affine
nib.save(nib.Nifti1Image(gqi_odf, affine), 'gqi_odf_norm.nii.gz')
nib.save(nib.Nifti1Image(gqi_gfa, affine), 'gfa_norm.nii.gz')

import numpy as np


np.savetxt('sphere724.txt', sphere.vertices)

