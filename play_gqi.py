import nibabel as nib
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.data import get_sphere
from dipy.viz.mayavi.spheres import show_odfs
from load_data import get_train_dsi


data, affine, gtab = get_train_dsi(30)

data = data[25-5:25+5, 25-5:25+5, 25]

gqi_model = GeneralizedQSamplingModel(gtab,
                                      method='gqi2',
                                      sampling_length=2,
                                      normalize_peaks=False)

gqi_fit = gqi_model.fit(data)

sphere = get_sphere('symmetric724')

gqi_odf = gqi_fit.odf(sphere)

show_odfs(gqi_odf[:, None], sphere)

