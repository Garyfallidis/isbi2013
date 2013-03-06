import nibabel as nib
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.dsi import DiffusionSpectrumDeconvModel
from dipy.data import get_sphere
from dipy.viz.mayavi.spheres import show_odfs
from load_data import get_train_dsi


data, affine, gtab = get_train_dsi(30)

gqi_model = GeneralizedQSamplingModel(gtab,
                                      method='gqi2',
                                      sampling_length=3,
                                      normalize_peaks=False)

gqi_fit = gqi_model.fit(data)
sphere = get_sphere('symmetric724')
gqi_odf = gqi_fit.odf(sphere)

import nibabel as nib

nib.save(nib.Nifti1Image(gqi_odf, affine), 'gqi_odf.nii.gz')

import numpy as np

np.savetxt('sphere724.txt', sphere.vertices)