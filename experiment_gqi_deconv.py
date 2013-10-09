import numpy as np
import nibabel as nib

from dipy.reconst.dti import fractional_anisotropy
from dipy.reconst.dti import TensorModel
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.dsi import DiffusionSpectrumDeconvModel
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, ConstrainedSDTModel
from dipy.data import get_sphere
from dipy.viz.mayavi.spheres import show_odfs
from dipy.reconst.shm import sf_to_sh

from load_data import get_train_dsi, get_train_rois, get_train_mask, get_train_hardi, get_train_dti
from show_streamlines import show_streamlines
from conn_mat import connectivity_matrix

from dipy.io.pickles import save_pickle, load_pickle

from time import time


if __name__ == '__main__':

    data, affine, gtab = get_train_hardi(10, denoised=None, Coupe=None)
    mask, affine = get_train_mask()

    tenmodel = TensorModel(gtab)

    tenfit = tenmodel.fit(data, mask)

    FA = fractional_anisotropy(tenfit.evals)

    FA[np.isnan(FA)] = 0

    indices = np.where(FA > 0.7)

    lambdas = tenfit.evals[indices][:, :2]

    S0s = data[indices][:, 0]

    S0 = np.mean(S0s)

    if S0 == 0 :
        S0 = 1
        
    l01 = np.mean(lambdas, axis = 0) 

    evals = np.array([l01[0], l01[1], l01[1]])

    print evals, l01[1] / l01[0], S0

    ratio = l01[1] / l01[0]
    #ratio = 0.2

    from dipy.data import get_sphere
    sphere = get_sphere('symmetric724')

#     csd_model = ConstrainedSphericalDeconvModel(gtab, (evals, S0), regul_sphere = sphere)
#     csd_fit = csd_model.fit(data[25 - 10:25 + 10, 25 - 10:25 + 10, 25])
#     csd_odf = csd_fit.odf(sphere)

#     from dipy.viz import fvtk
#     r = fvtk.ren()
#     fvtk.add(r, fvtk.sphere_funcs(csd_odf, sphere))
#     fvtk.show(r)
    

    csdt_model = ConstrainedSDTModel(gtab, ratio, regul_sphere = sphere)
    csdt_fit = csdt_model.fit(data[25 - 10:25 + 10, 25 - 10:25 + 10, 25])
    csdt_odf = csdt_fit.odf(sphere)

    from dipy.viz import fvtk
    r = fvtk.ren()
    fvtk.clear(r)
    fvtk.add(r, fvtk.sphere_funcs(csdt_odf, sphere))
    fvtk.show(r)




