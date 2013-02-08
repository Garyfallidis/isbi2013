import numpy as np
from dipy.data import get_data, dsi_voxels
from dipy.sims.voxel import MultiTensor, SticksAndBall
from dipy.core.sphere import Sphere
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from load_data import get_train_dsi
from pso import basic_pso, B_N_pso, B_N_H_pso
from dipy.core.sphere_stats import angular_similarity
from copy import deepcopy
from dipy.reconst.gqi import GeneralizedQSamplingModel
#from dipy.core.subdivide_octahedron import create_unit_sphere
# from dipy.viz.mayavi.spheres import show_odfs


sphere = get_sphere('symmetric724')
sphere = sphere.subdivide(1)

#print(sphere.vertices.shape)
#sphere2 = create_unit_sphere(5)


data, affine, gtab_full = get_train_dsi(30)

gtab = deepcopy(gtab_full)

bmin = 2000
bmax = 5000
gtab.b0s_mask = gtab.b0s_mask[(gtab.bvals >= bmin) & (gtab.bvals <= bmax)]
gtab.bvecs = gtab.bvecs[(gtab.bvals >= bmin) & (gtab.bvals <= bmax)]
gtab.bvals = gtab.bvals[(gtab.bvals >= bmin) & (gtab.bvals <= bmax)]

SNR = None

print('SNR = {} with {} gradients direction'.format(SNR,gtab.bvals.shape[0]))


# if angle = [rot1,rot2], you start with somethign aligned on Z you then rotate it
# around the Y axis by rot1 and then you rotate it around the Z axis by rot2
# S, sticks = SticksAndBall(gtab, d = 0.0015, S0=100, angles=[(45, 40), (45, 70)],
#                  fractions=[50, 50], snr=SNR)

mevals = np.array([[0.0021, 0.0002, 0.0002], [0.0021, 0.0002, 0.0002], [0.0021, 0.0002, 0.0002]])
S, sticks = MultiTensor(gtab, mevals, S0=100, angles=[(20, 10), (70, 20), (45, 60)],
                        fractions=[33.3, 33.3, 33.4], snr=SNR)

SS, stickss = MultiTensor(gtab_full, mevals, S0=100, angles=[(20, 10), (70, 20), (45, 60)],
                        fractions=[33.3, 33.3, 33.4], snr=SNR)

# more weigt on mins and max
nb_min_max = 6
min_max_weigt = 10
idx = np.argsort(S)
weigt = np.ones_like(S)
weigt[idx[:nb_min_max]] = min_max_weigt
weigt[idx[-nb_min_max:]] = min_max_weigt


def fit_quality_snb(S_gt,wts,lam, gtab, d=0.0015, angles=[(0, 0), (90, 0)],
                    fractions=[50, 50]):
    S, sticks = SticksAndBall(gtab, d, 100, angles, fractions, None)
    #return (wts*(np.abs(S - S_gt))).sum()
    return (wts*((S-S_gt)**2)).sum()
    #return ((S-S_gt)**2).sum() + lam * (wts*np.abs(S - S_gt)).sum()


# def metric_for_pso(pm):
#     return fit_quality_snb(S, gtab, pm[0], angles=[(pm[1], pm[2]), (pm[3], pm[4])],
#                            fractions=[50, 100 - 50])

def metric_for_pso(pm):
    return fit_quality_snb(S,weigt,1., gtab, pm[0], angles=[(pm[1], pm[2]), (pm[3], pm[4]), (pm[5], pm[6])],
                           fractions=[33.3, 33.3, 33.4])


bounds = np.array([[0.0001, 0.009], [0, 90], [0, 90], [0, 90], [0, 90], [0, 90], [0, 90]])  # ,[25, 75]])

# #check best AS at each todo/ boundarie also
# pm, fitnessValue = basic_pso(metric_for_pso, 100, 7, bounds, 50, .5, .5, .5, 1)


def metric_for_pso_B_N(pm):
    pmm = bounds[:,0] + (bounds[:,1]-bounds[:,0])*pm
    return fit_quality_snb(S,weigt,1., gtab, pmm[0], angles=[(pmm[1], pmm[2]), (pmm[3], pmm[4]), (pmm[5], pmm[6])],
                           fractions=[33.3, 33.3, 33.4])




#check best AS at each todo/ boundarie also
pmm, fitnessValue = B_N_pso(metric_for_pso_B_N, 100, 7, 50, .9, .9, .9, 0, 1)

# pmmH, fitnessValueH = B_N_H_pso(metric_for_pso_B_N, 50, 7, 25, .5, .5, .5, 1)

pm = bounds[:,0] + (bounds[:,1]-bounds[:,0])*pmm
# pmH = bounds[:,0] + (bounds[:,1]-bounds[:,0])*pmmH

S2, sticks2 = SticksAndBall(gtab, pm[0], angles=[(pm[1], pm[2]), (pm[3], pm[4]), (pm[5], pm[6])],
                            fractions=[33.3, 33.3, 33.4])

# S2H, sticks2H = SticksAndBall(gtab, pmH[0], angles=[(pmH[1], pmH[2]), (pmH[3], pmH[4]), (pmH[5], pmH[6])],
#                             fractions=[33.3, 33.3, 33.4])

print angular_similarity(sticks, sticks2)
# print angular_similarity(sticks, sticks2H)


gq = GeneralizedQSamplingModel(gtab_full,sampling_length=3.5)

gq.direction_finder.config(sphere=sphere, min_separation_angle=15,
                           relative_peak_threshold=.35)

gqfit = gq.fit(SS)
# gqodf = gqfit.odf(sphere)

gqdir = gqfit.directions

print angular_similarity(sticks, gqdir)


from dipy.viz import fvtk

r = fvtk.ren()

fvtk.add(r, fvtk.line(np.array([-sticks[0], sticks[0]]), fvtk.red))
fvtk.add(r, fvtk.line(np.array([-sticks[1], sticks[1]]), fvtk.red))
fvtk.add(r, fvtk.line(np.array([-sticks[2], sticks[2]]), fvtk.red))

fvtk.add(r, fvtk.line(np.array([-sticks2[0], sticks2[0]]), fvtk.blue))
fvtk.add(r, fvtk.line(np.array([-sticks2[1], sticks2[1]]), fvtk.blue))
fvtk.add(r, fvtk.line(np.array([-sticks2[2], sticks2[2]]), fvtk.blue))

for i in range(gqdir.shape[0]):
    fvtk.add(r, fvtk.line(np.array([-gqdir[i], gqdir[i]]), fvtk.green))
# fvtk.add(r, fvtk.line(np.array([[0, 0, 0], sticks2H[1]]), fvtk.green))
# fvtk.add(r, fvtk.line(np.array([[0, 0, 0], sticks2H[2]]), fvtk.green))

fvtk.show(r)

# from pylab import plot, show

# plot(S, 'b')
# plot(S2, 'r')
# show()
# # plot(np.abs(S - S2))
# plot((S - S2))
# show()
