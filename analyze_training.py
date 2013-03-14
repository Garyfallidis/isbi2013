

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
from dipy.data import get_sphere

from show_streamlines import show_streamlines
from conn_mat import connectivity_matrix

from dipy.io.pickles import save_pickle, load_pickle

from load_data import get_specific_data, get_train_mask
from reconst_to_conmat import pipe, streams_to_connmat

from time import time

training_yn = [True, False]
categories = ['dti', 'hardi', 'dsi']
snrs = [10, 20, 30]
odf_deconvs = [True, False]
total_variations = [True, False]
denoised_data = [True, False]

dname = 'data/'
dres = 'results/'


def create_file_prefix(training, category, snr, denoised, odf_deconv, tv, method):

    if training:
        filename = 'train'
    else:
        filename = 'test'
    filename += '_' + str(category) + '_snr_' + str(snr) + '_denoised_'
    filename += str(int(denoised)) + '_odeconv_' + str(int(odf_deconv))
    filename += '_tv_' + str(int(tv)) + '__' + method

    return filename


def csd(training, category, snr, denoised, odeconv, tv, method):

    data, affine, gtab = get_specific_data(training,
                                           category,
                                           snr,
                                           denoised)

    prefix = create_file_prefix(training,
                                category,
                                snr,
                                denoised,
                                odeconv,
                                tv,
                                method)

    mask, affine = get_train_mask()

    tenmodel = TensorModel(gtab)

    tenfit = tenmodel.fit(data, mask)

    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0

    mask[FA <= 0.1] = 0
    mask[FA > 1.] = 0

    indices = np.where(FA > 0.7)
    lambdas = tenfit.evals[indices][:, :2]
    S0s = data[indices][:, 0]
    S0 = np.mean(S0s)

    if S0==0:
        S0 = 1
    
    l01 = np.mean(lambdas, axis=0)
    
    evals = np.array([l01[0], l01[1], l01[1]])

    print evals, S0

    if category == 'dti':
        csd_model = ConstrainedSphericalDeconvModel(gtab, (evals, S0), sh_order=6)

    if category == 'hardi':
        csd_model = ConstrainedSphericalDeconvModel(gtab, (evals, S0), sh_order=8)

    csd_fit = csd_model.fit(data, mask)

    sphere = get_sphere('symmetric724')
    
    odf = csd_fit.odf(sphere)

    if tv == True:

        odf = tv_denoise_4d(odf, weight=0.1)

    nib.save(nib.Nifti1Image(odf, affine), dres + prefix + 'odf.nii.gz')

    odf_sh = sf_to_sh(odf, sphere, sh_order=8, basis_type='mrtrix')

    nib.save(nib.Nifti1Image(odf_sh, affine), dres + prefix + 'odf_sh.nii.gz')

    if training == True:
        return training_check(dres, prefix)


def training_check(dres, prefix)

    seeds_per_vox = 5

    num_of_cpus = 6

    cmd = 'python ~/Devel/scilpy/scripts/stream_local.py -odf ' + dres + prefix + 'odf_sh.nii.gz -m data/training-data_mask.nii.gz -s data/training-data_rois.nii.gz -n -' + str(seeds_per_vox) + ' -process ' + str(num_of_cpus) + ' -o ' + dres + prefix + 'streams.trk'
   
    pipe(cmd)

    mat, conn_mats, diffs = streams_to_connmat(dres + prefix + 'streams.trk', seeds_per_vox)

    save_pickle(dres + prefix + 'conn_mats.pkl', {'mat': mat, 'conn_mats': conn_mats, 'diffs': diffs})

    print dres + prefix + 'conn_mats.pkl'

    return diffs


def gqi(training, category, snr, denoised, odeconv, tv, method):
    pass


def show_odf_sample(filename):

    odf = nib.load(filename).get_data()

    sphere = get_sphere('symmetric724')
    from dipy.viz import fvtk
    r = fvtk.ren()
    #fvtk.add(r, fvtk.sphere_funcs(odf[:, :, 25], sphere))
    fvtk.add(r, fvtk.sphere_funcs(odf[25-10:25+10, 25-10:25+10, 25], sphere))
    fvtk.show(r)


def show_conn_mats(filename):
    d = load_pickle(filename)
    return d['mat'], d['conn_mats'], d['diffs']


if __name__ == '__main__':
   

    t0 = time()

    #diffs = csd(training=True, category='dti', snr=10, denoised=False, odeconv=False, tv=False, method='csd_')
    diffs = csd(training=True, category='dti', snr=30, denoised=False, odeconv=False, tv=False, method='csd_')

    print time() - t0
