training = [True, False]

categories = ['dti', 'hardi', 'dsi']

snrs = [10, 20, 30]

odf_deconvs = [True, False]

total_variations = [True, False]

denoised_data = [True, False]

dname = 'data/'


import numpy as np
import nibabel as nib

from dipy.reconst.dti import fractional_anisotropy
from dipy.reconst.dti import TensorModel
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.dsi import DiffusionSpectrumDeconvModel
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.csdeconv import ConstrainedSDTModel
from dipy.data import get_sphere
from dipy.viz.mayavi.spheres import show_odfs
from dipy.reconst.shm import sf_to_sh

from show_streamlines import show_streamlines
from conn_mat import connectivity_matrix

from dipy.io.pickles import save_pickle, load_pickle

from load_data import get_specific_data, get_train_mask


# def csd(training, categories, snrs, denoised):

data, affine, gtab = get_specific_data(training[0],
                                       categories[0],
                                       snrs[0],
                                       denoised_data[1])

def create_file_prefix(training, category, snr, denoised, odf_deconv, tv, method):

    if training:
        filename = 'train'
    else:
        filename = 'test'
    filename += '_' + str(category) + '_snr_' + str(snr) + '_denoised_' 
    filename += str(int(denoised)) + '_odeconv_' + str(int(odf_deconv))
    filename += '_tv_' + str(int(tv)) + '__' + method

    return filename

print create_file_prefix(training[1],
                         categories[0],
                         snrs[0],
                         denoised_data[0],
                         False,
                         False,
                         'csd_8')

1/0


mask, affine = get_train_mask()

tenmodel = TensorModel(gtab)

tenfit = tenmodel.fit(data, mask)

FA = fractional_anisotropy(tenfit.evals)

FA[np.isnan(FA)] = 0

mask[FA <= 0.1] = 0

indices = np.where(FA > 0.7)

lambdas = tenfit.evals[indices][:, :2]

S0s = data[indices][:, 0]

S0 = np.mean(S0s)

l01 = np.mean(lambdas, axis=0)

evals = np.array([l01[0], l01[1], l01[1]])

csd_model = ConstrainedSphericalDeconvModel(gtab, (evals, S0))

csd_fit = csd_model.fit(data, mask)

from dipy.data import get_sphere

sphere = get_sphere('symmetric724')

odf = csd_fit.odf(sphere)

#fname = 

nib.save(nib.Nifti1Image(odf, affine), 'odf.nii.gz')

odf_sh = sf_to_sh(odf, sphere, sh_order=8,
                  basis_type='mrtrix')

nib.save(nib.Nifti1Image(odf_sh, affine), 'odf_sh.nii.gz')

from dipy.viz import fvtk
r = fvtk.ren()
fvtk.add(r, fvtk.sphere_funcs(odf[:, :, 25], sphere))
fvtk.show(r)
