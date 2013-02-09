import nibabel as nib
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.dsi import DiffusionSpectrumDeconvModel
from dipy.data import get_sphere
from load_data import (get_train_dsi,
                       get_train_hardi,
                       get_train_dti)
from dipy.sims.voxel import SticksAndBall, MultiTensor

#data, affine, gtab = get_train_hardi(30)

mevals = np.array(([0.0015, 0.0003, 0.0003],
                   [0.0015, 0.0003, 0.0003]))

sphere = get_sphere('symmetric362')
#sphere = sphere.subdivide(1)

from dipy.core.gradients import gradient_table

gtab = gradient_table(4000 * np.ones(len(sphere.vertices)),
                      sphere.vertices)

S, sticks = MultiTensor(gtab,
                        mevals, S0=100,
                        angles=[(0, 0), (30, 0)],
                        fractions=[50, 50],
                        snr=None)

S2, sticks2 = MultiTensor(gtab,
                          mevals, S0=100,
                          angles=[(0, 0), (90, 0)],
                          fractions=[50, 50],
                          snr=None)


from dipy.viz import fvtk

r = fvtk.ren()
# fvtk.add(r, fvtk.dots(sphere.vertices, fvtk.red))
fvtk.add(r, fvtk.sphere_funcs(np.vstack((S, S2)), 
		 sphere, scale = 30., norm=False))
fvtk.add(r, fvtk.sphere_funcs(np.abs(S-S2), 
		 sphere, scale = 30., norm=False))

fvtk.show(r)
