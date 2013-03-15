import numpy as np
import nibabel as nib

from subprocess import Popen, PIPE

from dipy.reconst.dti import fractional_anisotropy
from dipy.reconst.dti import TensorModel
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.dsi import DiffusionSpectrumDeconvModel
from dipy.data import get_sphere
#from dipy.viz.mayavi.spheres import show_odfs
from dipy.reconst.shm import sf_to_sh

from load_data import get_train_dsi, get_train_rois, get_train_mask
from show_streamlines import show_streamlines
from conn_mat import connectivity_matrix
from total_variation import tv_denoise_4d

from dipy.io.pickles import save_pickle, load_pickle

from time import time


def pipe(cmd):
    """ A tiny pipeline system to run external tools.

    For more advanced pipelining use nipype http://www.nipy.org/nipype
    """
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    sto = p.stdout.readlines()
    ste = p.stderr.readlines()
    #print(sto)
    #print(ste)


def streams_to_connmat(filename, seeds_per_voxel=1, thr=[0.25, 0.5, 0.75]):

    streams, hdr = nib.trackvis.read(filename)
    streamlines = [s[0] for s in streams]

    # show_streamlines(streamlines, opacity=0.5)

    rois, affine = get_train_rois()

    mat, srois, ratio = connectivity_matrix(streamlines, rois)

    golden_mat = np.load('train_connmat.npy')

    lr = []
    for i in range(1, 41):
        lr.append(np.sum(rois == i))

    lr = seeds_per_voxel * np.array(lr, dtype='f8')

    mat /= lr

    mat += mat.T

    golden_mat += golden_mat.T

    conn_mats = []
    diffs = []
    for th in thr:
        conn_mat = mat > th
        conn_mats.append(conn_mat)
        diffs.append(np.sum(np.abs(conn_mat - golden_mat)))

    return mat, conn_mats, diffs



if __name__ == '__main__':

    data, affine, gtab = get_train_dsi(30)
    mask, affine = get_train_mask()

    tenmodel = TensorModel(gtab)

    tenfit = tenmodel.fit(data, mask)

    FA = fractional_anisotropy(tenfit.evals)

    FA[np.isnan(FA)] = 0

    indices = np.where(FA > 0.7)

    lambdas = tenfit.evals[indices][:, :2]

    S0s = data[indices][:, 0]

    S0 = np.mean(S0s)

    l01 = np.mean(lambdas, axis = 0) 

    evals = np.array([l01[0], l01[1], l01[1]])

    #1/0

    data = data[25 - 10:25 + 10, 25 - 10:25 + 10, 25]
    mask = mask[25 - 10:25 + 10, 25 - 10:25 + 10, 25]
    # data = data[:, :, 25]

    model_tag = 'dsdeconv_'

    model = GeneralizedQSamplingModel(gtab,
                                      method='gqi2',
                                      sampling_length=3.,
                                      normalize_peaks=False)

    #model = DiffusionSpectrumDeconvModel(gtab)

    fit = model.fit(data, mask)

    sphere = get_sphere('symmetric724')

    odf = fit.odf(sphere)

    


    #nib.save(nib.Nifti1Image(odf, affine), model_tag + 'odf.nii.gz')

    odf_sh = sf_to_sh(odf, sphere, sh_order=8,
                      basis_type='mrtrix')

    #nib.save(nib.Nifti1Image(odf_sh, affine), model_tag + 'odf_sh.nii.gz')

    from dipy.reconst.csdeconv import odf_sh_to_sharp

    reg_sphere = get_sphere('symmetric724')

    fodf_sh = odf_sh_to_sharp(odf_sh,
                              reg_sphere, basis='mrtrix', ratio=3.8 / 16.6,
                              sh_order=8, Lambda=1., tau=1.)

    # nib.save(nib.Nifti1Image(odf_sh, affine), model_tag + 'fodf_sh.nii.gz')

    from dipy.reconst.shm import real_sph_harm_mrtrix
    from dipy.data import get_sphere
    from dipy.core.geometry import cart2sphere

    r, theta, phi = cart2sphere(sphere.x, sphere.y, sphere.z)
    B_regul, m, n = real_sph_harm_mrtrix(8, theta[:, None], phi[:, None])

    fodf = np.dot(fodf_sh, B_regul.T)

    from dipy.viz import fvtk

    #odf = odf[25 - 10:25 + 10, 25 - 10:25 + 10, 25]
    r = fvtk.ren()
    fvtk.add(r, fvtk.sphere_funcs(odf, sphere))
    fvtk.show(r)
    fvtk.clear(r)

    # odf_var = tv_denoise_4d(odf, weight=0.1)
    # fvtk.add(r, fvtk.sphere_funcs(odf_var, sphere))
    # fvtk.show(r)
    # fvtk.clear(r)

    # #odf_sh2 = odf_sh[25 - 10:25 + 10, 25 - 10:25 + 10, 25]
    # odf2 = np.dot(odf_sh, B_regul.T)
    # fvtk.add(r, fvtk.sphere_funcs(odf2, sphere))
    # fvtk.show(r)
    # fvtk.clear(r)

    r = fvtk.ren()
    fvtk.add(r, fvtk.sphere_funcs(fodf, sphere))
    fvtk.show(r)

    seeds_per_vox = 5
    num_of_cpus = 6

    # cmd = 'python ~/Devel/scilpy/scripts/stream_local.py -odf ' + model_tag + 'odf_sh.nii.gz -m data/training-data_mask.nii.gz -s data/training-data_rois.nii.gz -n -' + \
    #        str(seeds_per_vox) +' -process ' + str(num_of_cpus) + ' -o ' + model_tag + 'streams.trk -maximum'
    # pipe(cmd)

    # mat, conn_mats, diffs = streams_to_connmat(model_tag + 'streams.trk', 9)

    # save_pickle(model_tag + 'conn_mats.pkl', {'mat':mat, 'conn_mats':conn_mats, 'diffs':diffs})
