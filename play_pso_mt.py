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
# from dipy.reconst.gqi import GeneralizedQSamplingModel
#from dipy.core.subdivide_octahedron import create_unit_sphere
# from dipy.viz.mayavi.spheres import show_odfs


sphere = get_sphere('symmetric724')
sphere = sphere.subdivide(1)

#print(sphere.vertices.shape)
#sphere2 = create_unit_sphere(5)


data, affine, gtab_full = get_train_dsi(30)

gtab = deepcopy(gtab_full)

#subset of dsi gtab
bmin = 1500
bmax = 4000
gtab.b0s_mask = gtab.b0s_mask[(gtab.bvals >= bmin) & (gtab.bvals <= bmax)]
gtab.bvecs = gtab.bvecs[(gtab.bvals >= bmin) & (gtab.bvals <= bmax)]
gtab.bvals = gtab.bvals[(gtab.bvals >= bmin) & (gtab.bvals <= bmax)]
NN = gtab.bvals.shape[0]

SNR = 30.

print('SNR = {} with {} gradients direction ({}-{})'.format(SNR,gtab.bvals.shape[0],bmin,bmax))


# if angle = [rot1,rot2], you start with somethign aligned on Z you then rotate it
# around the Y axis by rot1 and then you rotate it around the Z axis by rot2
# S, sticks = SticksAndBall(gtab, d = 0.0015, S0=100, angles=[(45, 40), (45, 70)],
#                  fractions=[50, 50], snr=SNR)


#equal part shard 3 tensor crossing
mevals = np.array([[0.0017, 0.0003, 0.0003], [0.0017, 0.0003, 0.0003], [0.0017, 0.0003, 0.0003]])
ang=[(20, 10), (70, 20), (45, 60)]
S, sticks = MultiTensor(gtab, mevals, S0=100, angles=ang,
                        fractions=[33.3, 33.3, 33.4], snr=SNR)

# SS, stickss = MultiTensor(gtab_full, mevals, S0=100, angles=ang,
#                         fractions=[33.3, 33.3, 33.4], snr=SNR

#noiseless signal
SS, stickss = MultiTensor(gtab, mevals, S0=100, angles=ang,
                        fractions=[33.3, 33.3, 33.4], snr=None)

# more weigt on mins and max
# nb of mins and maxs weigted
nb_min_max = 6
# weigt increase
min_max_weigt = 1
idx = np.argsort(S)
weigt = np.ones_like(S)
weigt[idx[:nb_min_max]] = min_max_weigt
weigt[idx[-nb_min_max:]] = min_max_weigt
print('weigt = {} for {}'.format(min_max_weigt,nb_min_max))

def fit_quality_mt(S_gt,wts,lam, gtab,mevalss, angles=[(0, 0), (90, 0)],
                    fractions=[50, 50]):
    S, sticks = MultiTensor(gtab, mevalss, 100, angles,fractions, None)
    # return (wts*(np.abs(S - S_gt))).sum()
    return (wts*((S-S_gt)**2)).sum()
    #return ((S-S_gt)**2).sum() + lam * (wts*np.abs(S - S_gt)).sum()


def metric_for_pso(pm):
    return fit_quality_mt(S,weigt,1., gtab, np.array([[pm[0],pm[1],pm[1]],[pm[2],pm[3],pm[3]],[pm[4],pm[5],pm[5]]]), angles=[(pm[6], pm[7]), (pm[8], pm[9]), (pm[10], pm[11])],fractions=[33.3, 33.3, 33.4])


bounds = np.array([[0.001,0.003],[0.0001,0.0005],[0.001,0.003],[0.0001,0.0005],[0.001,0.003],[0.0001,0.0005], [0, 90], [0, 90], [0, 90], [0, 90], [0, 90], [0, 90]])  # ,[25, 75]])
# bounds = np.array([[0.0016,0.0018],[0.0002,0.0004],[0.0016,0.0018],[0.0002,0.0004],[0.0016,0.0018],[0.0002,0.0004], [0, 90], [0, 90], [0, 90], [0, 90], [0, 90], [0, 90]])  # ,[25, 75]])
# bounds = np.array([[0.0001,0.01],[0.0001,0.01],[0.0001,0.01],[0.0001,0.01],[0.0001,0.01],[0.0001,0.01], [0, 90], [0, 90], [0, 90], [0, 90], [0, 90], [0, 90]])  # ,[25, 75]])


def metric_for_pso_B_N(pm):
    pmm = bounds[:,0] + (bounds[:,1]-bounds[:,0])*pm
    return fit_quality_mt(S,weigt,1., gtab, np.array([[pmm[0],pmm[1],pmm[1]],[pmm[2],pmm[3],pmm[3]],[pmm[4],pmm[5],pmm[5]]]), angles=[(pmm[6], pmm[7]), (pmm[8], pmm[9]), (pmm[10], pmm[11])],fractions=[33.3, 33.3, 33.4])

#reset particule position if it get stuck for long
soft_reset = 1
npart=100
niter=100
truc = SNR
if truc==None:
    truc = np.inf
#stopped if the metric get below good_enough
good_enough = 0 #50 + (0.8*(100/truc))**2*NN
# good_enough=((NN+nb_min_max*(min_max_weigt-1)*(100/truc)**2)
# good_enough= NN*(100/truc)
print(npart,soft_reset,niter)

print(good_enough,fit_quality_mt(S,weigt,1., gtab, mevals, ang,fractions=[33.3, 33.3, 33.4]))


for i in range(5):
    pmm, fV = B_N_pso(metric_for_pso_B_N, npart, 12, niter, 0.75, 0.75, 0.75,soft_reset,good_enough, 1)



    pm = bounds[:,0] + (bounds[:,1]-bounds[:,0])*pmm

    S2, sticks2 = MultiTensor(gtab, np.array([[pm[0],pm[1],pm[1]],[pm[2],pm[3],pm[3]],[pm[4],pm[5],pm[5]]]), angles=[(pm[6], pm[7]), (pm[8], pm[9]), (pm[10], pm[11])],
                               fractions=[33.3, 33.3, 33.4])

    print angular_similarity(sticks, sticks2)


# gq = GeneralizedQSamplingModel(gtab_full,sampling_length=3.5)

# gq.direction_finder.config(sphere=sphere, min_separation_angle=15,
#                            relative_peak_threshold=.35)

# gqfit = gq.fit(SS)
# gqodf = gqfit.odf(sphere)

# gqdir = gqfit.directions

# print angular_similarity(sticks, gqdir)


from dipy.viz import fvtk

r = fvtk.ren()

fvtk.add(r, fvtk.line(np.array([-sticks[0], sticks[0]]), fvtk.red))
fvtk.add(r, fvtk.line(np.array([-sticks[1], sticks[1]]), fvtk.red))
fvtk.add(r, fvtk.line(np.array([-sticks[2], sticks[2]]), fvtk.red))

fvtk.add(r, fvtk.line(np.array([-sticks2[0], sticks2[0]]), fvtk.blue))
fvtk.add(r, fvtk.line(np.array([-sticks2[1], sticks2[1]]), fvtk.blue))
fvtk.add(r, fvtk.line(np.array([-sticks2[2], sticks2[2]]), fvtk.blue))

# for i in range(gqdir.shape[0]):
#     fvtk.add(r, fvtk.line(np.array([-gqdir[i], gqdir[i]]), fvtk.green))
# # fvtk.add(r, fvtk.line(np.array([[0, 0, 0], sticks2H[1]]), fvtk.green))
# # fvtk.add(r, fvtk.line(np.array([[0, 0, 0], sticks2H[2]]), fvtk.green))

fvtk.show(r)

# from pylab import plot, show

# plot(S, 'b')
# plot(S2, 'r')
# show()
# # plot(np.abs(S - S2))
# plot((S - S2))
# show()
