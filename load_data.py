import nibabel as nib
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.data import get_sphere
from dipy.viz.mayavi.spheres import show_odfs

dname = 'data/'

fdsi_raw = dname + 'DWIS_dsi-scheme_SNR-30.nii.gz'
fdsi_bvals = dname + 'dsi-scheme.bval'
fdsi_bvecs = dname + 'dsi-scheme.bvec'

bvals, bvecs = read_bvals_bvecs(fdsi_bvals, fdsi_bvecs)
gtab = gradient_table(bvals, bvecs)
img = nib.load(fdsi_raw)

data = img.get_data()

gqi_model = GeneralizedQSamplingModel(gtab,
                                      method='gqi2',
                                      sampling_length=2,
                                      normalize_peaks=False)

gqi_fit = gqi_model.fit(data)

sphere = get_sphere('symmetric724')

gqi_odf = gqi_fit.odf(sphere)

#show_odfs(gqi_odf[25-7:25+7, 25-7:25+7, 21, None], sphere)
show_odfs(gqi_odf[25-5:25+5, 25-5:25+5, 25][:, None], sphere)
