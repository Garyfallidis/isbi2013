#from dipy.reconst.tests.test_dsi import sticks_and_ball_dummies
from dipy.sims.voxel import SticksAndBall
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.data import get_sphere
from dipy.viz.mayavi.spheres import show_odfs
import numpy as np

dname = 'data/'

fdsi_raw = dname + 'DWIS_dsi-scheme_SNR-30.nii.gz'
fdsi_bvals = dname + 'dsi-scheme.bval'
fdsi_bvecs = dname + 'dsi-scheme.bvec'

bvals, bvecs = read_bvals_bvecs(fdsi_bvals, fdsi_bvecs)
gtab = gradient_table(bvals, bvecs)


#sb = sticks_and_ball_dummies(gtab)

S, sticks = SticksAndBall(gtab, d=0.0015, S0=100,
                              angles=[(0, 0), (30, 0)],
                              fractions=[50, 50], snr=30.0)

sphere = get_sphere('symmetric724')
sphere = sphere.subdivide(n=1)

r=np.zeros(sphere.vertices.shape[0])
sampling_lengths=[1.25,1.5,2.,2.5,3.,3.5,4.]
for ss in sampling_lengths:

	gqi_model = GeneralizedQSamplingModel(gtab,
	                                      method='gqi2',
	                                      sampling_length=ss,
	                                      normalize_peaks=False)

	gqi_fit = gqi_model.fit(S)

	gqi_odf = gqi_fit.odf(sphere)


	r=np.vstack((r,gqi_odf))

show_odfs(r[:,None,None], sphere)
