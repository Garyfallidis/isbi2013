import nibabel as nib
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.dsi import DiffusionSpectrumDeconvModel
from dipy.data import get_sphere
from dipy.viz.mayavi.spheres import show_odfs
from load_data import get_train_dsi
from dipy.sims.voxel import MultiTensor


data, affine, gtab = get_train_dsi(30)
# data = data[25 - 10:25 + 10, 25 - 10:25 + 10, 25]
# data = data[:, :, 25]


mevals = np.array([[0.0017, 0.0003, 0.0003],
                   [0.0017, 0.0003, 0.0003],
                   [0.0017, 0.0003, 0.0003]])

# ang = [(20, 10), (70, 20), (45, 60)]
ang = [(0, 0), (45, 45), (90, 90)]

data, sticks = MultiTensor(gtab, mevals, S0=100, angles=ang,
                           fractions=[33.3, 33.3, 33.4], snr=100)


gqi_model = GeneralizedQSamplingModel(gtab,
                                      method='gqi2',
                                      sampling_length=4.,
                                      normalize_peaks=False)

gqi_fit = gqi_model.fit(data)

sphere = get_sphere('symmetric724')
gqi_odf = gqi_fit.odf(sphere)


def optimal_transform(model, data, sphere):
    from dipy.reconst.gqi import squared_radial_component

    H = squared_radial_component

    b_vector = model.b_vector
    bnorm = np.sqrt(np.sum(b_vector ** 2, axis=1))

    # # linearize
    # b_vector2 = b_vector * bnorm[:, None]
    # b_vector2 /= b_vector.max()

    # # push higher q-space values
    # bnorm2 = np.sqrt(np.sum(b_vector2 ** 2, axis=1))
    # bnorm2n = np.interp(bnorm2, [bnorm2.min(), bnorm2.max()], [0, 1])
    # bpush = bnorm2n ** 4

    # b_vector3 = b_vector * bpush[:, None]

    proj = np.dot(b_vector, sphere.vertices.T)
    #proj[np.abs(proj) < 0.1] = 0
    #proj = proj * (bnorm[:, None] / 5.)


    gqi_vector = np.real(H(proj * model.Lambda / np.pi))

    return np.dot(data, gqi_vector), proj

odf, proj = optimal_transform(gqi_model, data, sphere)


from dipy.viz import fvtk

r = fvtk.ren()
fvtk.add(r, fvtk.sphere_funcs(gqi_odf, sphere))
fvtk.show(r)

dsi_model = DiffusionSpectrumDeconvModel(gtab)
dsi_odf = dsi_model.fit(data).odf(sphere)

fvtk.clear(r)
fvtk.add(r, fvtk.sphere_funcs(dsi_odf, sphere))
fvtk.show(r)

fvtk.clear(r)
fvtk.add(r, fvtk.sphere_funcs(odf, sphere))
fvtk.show(r)


def investigate_internals():

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
    plot(x, np.sinc(x), 'b', x, H(x), 'g')
    title(str(sampling_length))
