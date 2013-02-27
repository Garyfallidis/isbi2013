import nibabel as nib
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.dsi import DiffusionSpectrumDeconvModel
from dipy.data import get_sphere
from dipy.viz.mayavi.spheres import show_odfs
from load_data import get_train_dsi


data, affine, gtab = get_train_dsi(30)

data = data[25 - 10:25 + 10, 25 - 10:25 + 10, 25]
# data = data[:, :, 25]

gqi_model = GeneralizedQSamplingModel(gtab,
                                      method='gqi2',
                                      sampling_length=3,
                                      normalize_peaks=False)

gqi_fit = gqi_model.fit(data)
sphere = get_sphere('symmetric724')
gqi_odf = gqi_fit.odf(sphere)

from dipy.viz import fvtk
r = fvtk.ren()
fvtk.add(r, fvtk.sphere_funcs(gqi_odf, sphere))
fvtk.show(r)

dsi_model = DiffusionSpectrumDeconvModel(gtab)
dsi_odf = dsi_model.fit(data).odf(sphere)

fvtk.clear(r)
fvtk.add(r, fvtk.sphere_funcs(dsi_odf, sphere))
fvtk.show(r)


def bvl_min_max(b_vector, sphere, sampling_length):
    bv = np.dot(gqi_model.b_vector, sphere.vertices.T)
    bv.max()
    bv.min()
    bvl = bv * sampling_length / np.pi
    return bvl


def H(x):
    return (2 * x * np.cos(x) + (x ** 2 - 2) * np.sin(x)) / x ** 3

sampling_length = 1.2
bvl = bvl_min_max(gqi_model.b_vector, sphere, sampling_length)
x = np.linspace(bvl.min(), bvl.max(), 100)
figure(1)
plot(x, np.sinc(x), 'b', x, H(x), 'g')
title(str(sampling_length))

sampling_length = 3
bvl = bvl_min_max(gqi_model.b_vector, sphere, sampling_length)
x = np.linspace(bvl.min(), bvl.max(), 100)
figure(2)
plot(x, np.sinc(x), 'b+', x, H(x), 'g+')
title(str(sampling_length))
