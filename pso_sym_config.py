import numpy as np
from dipy.sims.voxel import MultiTensor, SticksAndBall
from dipy.core.sphere import Sphere
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from load_data import get_train_dsi
from pso import B_N_pso
from dipy.core.sphere_stats import angular_similarity
from copy import deepcopy


#reset particule position if it get stuck for long
soft_reset = 0
npart = 100
niter = 125
#stopped if the metric get below good_enough
good_enough = 0  # 50 + (0.8*(100/truc))**2*NN
# print(npart, soft_reset, niter)
plot_it = 0
verbo = 0


# sphere = get_sphere('symmetric724')
# sphere = sphere.subdivide(1)

# _, _, gtab_full = get_train_dsi(30)

# gtab = deepcopy(gtab_full)

# #subset of dsi gtab
# bmin = 1500
# bmax = 4000
# gtab.b0s_mask = gtab.b0s_mask[(gtab.bvals >= bmin) & (gtab.bvals <= bmax)]
# gtab.bvecs = gtab.bvecs[(gtab.bvals >= bmin) & (gtab.bvals <= bmax)]
# gtab.bvals = gtab.bvals[(gtab.bvals >= bmin) & (gtab.bvals <= bmax)]
# NN = gtab.bvals.shape[0]

# 0 : single shell
# 1 : half DSI
# 2 : 2 shells
scheme = 0

if scheme == 0:

    #Single shell
    bmin = 0
    bmax = 5000
    NN = 18
    f = open('../cs-dsi/data/directions/Elec0{}.txt'.format(NN))
    _ = f.readline()
    bvecs = np.loadtxt(f)
    f.close()
    bvecs = np.vstack((np.array([0., 0., 0.]), bvecs))
    bvals = bmax * np.ones(NN + 1)
    bvals[0] = 0
    gtab = gradient_table(bvals, bvecs)

elif scheme == 1:

    #DSI scheme
    qspace = []
    for ix in range(-5, 6):
        for iy in range(-5, 6):
            for iz in range(-5, 6):
                if ix ** 2 + iy ** 2 + iz ** 2 <= 25:
                    qspace.append([ix, iy, iz])
    qspace = np.array(qspace)

    #half DSI bvals and bvecs
    bmin = 0
    bmax = 8000
    bvals = []
    bvecs = []
    for ind in range(257):
        bvecs.append(qspace[ind] / np.sqrt((qspace[ind] ** 2).sum()))
        bvals.append((bmax / 25.) * (qspace[ind, :] ** 2).sum())
    bvecs.append(qspace[257])
    bvals.append(0)

    gtab = gradient_table(bvals, bvecs)


SNR = 30.

print('SNR = {} with {} gradients direction (b{}-b{})'.format(SNR, gtab.bvals.shape[0], bmin, bmax))


#Test all cases: 1-3 trueFiber with 1-3 estFiber in pso, all with equal angles (not in pso) and equal eVals, for tight and tight angles
mevals1 = np.array([0.0017, 0.0003, 0.0003])
mevals1 = mevals1[None, :]
mevals2 = np.array([[0.0017, 0.0003, 0.0003], [0.0017, 0.0003, 0.0003]])
mevals3 = np.array([[0.0017, 0.0003, 0.0003], [0.0017, 0.0003, 0.0003], [0.0017, 0.0003, 0.0003]])

frac1 = [100.]
frac2 = [50., 50.]
frac3 = [33.33, 33.33, 33.34]


ang1 = [(45, 45)]
ang2_easy = [(10, 10), (100, -10)]
ang3_easy = [(10, -10), (85, 10), (95, 80)]

ang2_tight = [(20, 10), (50, 20)]
ang3_tight = [(20, 10), (70, 20), (45, 60)]


S_1, sk_1 = MultiTensor(gtab, mevals1, S0=100, angles=ang1,
                        fractions=frac1, snr=SNR)
S_2_easy, sk_2_easy = MultiTensor(gtab, mevals2, S0=100, angles=ang2_easy,
                                  fractions=frac2, snr=SNR)
S_2_tight, sk_2_tight = MultiTensor(gtab, mevals2, S0=100, angles=ang2_tight,
                                    fractions=frac2, snr=SNR)
S_3_easy, sk_3_easy = MultiTensor(gtab, mevals3, S0=100, angles=ang3_easy,
                                  fractions=frac3, snr=SNR)
S_3_tight, sk_3_tight = MultiTensor(gtab, mevals3, S0=100, angles=ang3_tight,
                                    fractions=frac3, snr=SNR)


def fit_quality_mt(S_gt, gtab, mevalss, angles=[(0, 0), (90, 0)],
                   fractions=[50, 50]):
    S, sticks = MultiTensor(gtab, mevalss, 100, angles, fractions, None)
    return ((S - S_gt) ** 2).sum()


bounds1 = np.array([[0.001, 0.003], [0.0001, 0.0005], [0, 180], [-90, 90]])  # ,[25, 75]])
bounds2 = np.array([[0.001, 0.003], [0.0001, 0.0005], [0.001, 0.003], [0.0001, 0.0005], [0, 180], [-90, 90], [0, 180], [-90, 90]])  # ,[25, 75]])
bounds3 = np.array([[0.001, 0.003], [0.0001, 0.0005], [0.001, 0.003], [0.0001, 0.0005], [0.001, 0.003], [0.0001, 0.0005], [0, 180], [-90, 90], [0, 180], [-90, 90], [0, 180], [-90, 90]])  # ,[25, 75]])


def metric_for_pso_B_N_est1_1(pm):
    pmm = bounds1[:, 0] + (bounds1[:, 1] - bounds1[:, 0]) * pm
    return fit_quality_mt(S_1, gtab, np.array([[pmm[0], pmm[1], pmm[1]]]), angles=[(pmm[2], pmm[3])], fractions=frac1)


def metric_for_pso_B_N_est1_2_easy(pm):
    pmm = bounds1[:, 0] + (bounds1[:, 1] - bounds1[:, 0]) * pm
    return fit_quality_mt(S_2_easy, gtab, np.array([[pmm[0], pmm[1], pmm[1]]]), angles=[(pmm[2], pmm[3])], fractions=frac1)


def metric_for_pso_B_N_est1_2_tight(pm):
    pmm = bounds1[:, 0] + (bounds1[:, 1] - bounds1[:, 0]) * pm
    return fit_quality_mt(S_2_tight, gtab, np.array([[pmm[0], pmm[1], pmm[1]]]), angles=[(pmm[2], pmm[3])], fractions=frac1)


def metric_for_pso_B_N_est1_3_easy(pm):
    pmm = bounds1[:, 0] + (bounds1[:, 1] - bounds1[:, 0]) * pm
    return fit_quality_mt(S_3_easy, gtab, np.array([[pmm[0], pmm[1], pmm[1]]]), angles=[(pmm[2], pmm[3])], fractions=frac1)


def metric_for_pso_B_N_est1_3_tight(pm):
    pmm = bounds1[:, 0] + (bounds1[:, 1] - bounds1[:, 0]) * pm
    return fit_quality_mt(S_3_tight, gtab, np.array([[pmm[0], pmm[1], pmm[1]]]), angles=[(pmm[2], pmm[3])], fractions=frac1)


def metric_for_pso_B_N_est2_1(pm):
    pmm = bounds2[:, 0] + (bounds2[:, 1] - bounds2[:, 0]) * pm
    return fit_quality_mt(S_1, gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]]]), angles=[(pmm[4], pmm[5]), (pmm[6], pmm[7])], fractions=frac2)


def metric_for_pso_B_N_est2_2_easy(pm):
    pmm = bounds2[:, 0] + (bounds2[:, 1] - bounds2[:, 0]) * pm
    return fit_quality_mt(S_2_easy, gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]]]), angles=[(pmm[4], pmm[5]), (pmm[6], pmm[7])], fractions=frac2)


def metric_for_pso_B_N_est2_2_tight(pm):
    pmm = bounds2[:, 0] + (bounds2[:, 1] - bounds2[:, 0]) * pm
    return fit_quality_mt(S_2_tight, gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]]]), angles=[(pmm[4], pmm[5]), (pmm[6], pmm[7])], fractions=frac2)


def metric_for_pso_B_N_est2_3_easy(pm):
    pmm = bounds2[:, 0] + (bounds2[:, 1] - bounds2[:, 0]) * pm
    return fit_quality_mt(S_3_easy, gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]]]), angles=[(pmm[4], pmm[5]), (pmm[6], pmm[7])], fractions=frac2)


def metric_for_pso_B_N_est2_3_tight(pm):
    pmm = bounds2[:, 0] + (bounds2[:, 1] - bounds2[:, 0]) * pm
    return fit_quality_mt(S_3_tight, gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]]]), angles=[(pmm[4], pmm[5]), (pmm[6], pmm[7])], fractions=frac2)


def metric_for_pso_B_N_est3_1(pm):
    pmm = bounds3[:, 0] + (bounds3[:, 1] - bounds3[:, 0]) * pm
    return fit_quality_mt(S_1, gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]], [pmm[4], pmm[5], pmm[5]]]), angles=[(pmm[6], pmm[7]), (pmm[8], pmm[9]), (pmm[10], pmm[11])], fractions=frac3)


def metric_for_pso_B_N_est3_2_easy(pm):
    pmm = bounds3[:, 0] + (bounds3[:, 1] - bounds3[:, 0]) * pm
    return fit_quality_mt(S_2_easy, gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]], [pmm[4], pmm[5], pmm[5]]]), angles=[(pmm[6], pmm[7]), (pmm[8], pmm[9]), (pmm[10], pmm[11])], fractions=frac3)


def metric_for_pso_B_N_est3_2_tight(pm):
    pmm = bounds3[:, 0] + (bounds3[:, 1] - bounds3[:, 0]) * pm
    return fit_quality_mt(S_2_tight, gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]], [pmm[4], pmm[5], pmm[5]]]), angles=[(pmm[6], pmm[7]), (pmm[8], pmm[9]), (pmm[10], pmm[11])], fractions=frac3)


def metric_for_pso_B_N_est3_3_easy(pm):
    pmm = bounds3[:, 0] + (bounds3[:, 1] - bounds3[:, 0]) * pm
    return fit_quality_mt(S_3_easy, gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]], [pmm[4], pmm[5], pmm[5]]]), angles=[(pmm[6], pmm[7]), (pmm[8], pmm[9]), (pmm[10], pmm[11])], fractions=frac3)


def metric_for_pso_B_N_est3_3_tight(pm):
    pmm = bounds3[:, 0] + (bounds3[:, 1] - bounds3[:, 0]) * pm
    return fit_quality_mt(S_3_tight, gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]], [pmm[4], pmm[5], pmm[5]]]), angles=[(pmm[6], pmm[7]), (pmm[8], pmm[9]), (pmm[10], pmm[11])], fractions=frac3)


if plot_it:
    from dipy.viz import fvtk


#######################################
pm, fV = B_N_pso(metric_for_pso_B_N_est1_1, npart, 4, niter, 0.75, 0.75, 0.75, soft_reset, good_enough, verbo)

pmm = bounds1[:, 0] + (bounds1[:, 1] - bounds1[:, 0]) * pm

_, sk_est = MultiTensor(gtab, np.array([[pmm[0], pmm[1], pmm[1]]]), angles=[(pmm[2], pmm[3])], fractions=frac1)

as_est = angular_similarity(sk_est, sk_1)
print('Tru = {}, PSO = {}, Ang = {}. AS = {:.5f}. (#part = {}, #iter = {})'.format(1, 1, 'easy', as_est, npart, niter))

if plot_it:
    r = fvtk.ren()

    fvtk.add(r, fvtk.line(np.array([-sk_1[0], sk_1[0]]), fvtk.red))
    # fvtk.add(r, fvtk.line(np.array([-sk[1], sk[1]]), fvtk.red))
    # fvtk.add(r, fvtk.line(np.array([-sk[2], sk[2]]), fvtk.red))

    fvtk.add(r, fvtk.line(np.array([-sk_est[0], sk_est[0]]), fvtk.blue))
    # fvtk.add(r, fvtk.line(np.array([-sk[1], sk[1]]), fvtk.blue))
    # fvtk.add(r, fvtk.line(np.array([-sk[2], sk[2]]), fvtk.blue))

    fvtk.show(r)
#######################################
pm, fV = B_N_pso(metric_for_pso_B_N_est2_1, npart, 8, niter, 0.75, 0.75, 0.75, soft_reset, good_enough, verbo)

pmm = bounds2[:, 0] + (bounds2[:, 1] - bounds2[:, 0]) * pm

_, sk_est = MultiTensor(gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]]]), angles=[(pmm[4], pmm[5]), (pmm[6], pmm[7])], fractions=frac2)

as_est = angular_similarity(sk_est, sk_1)
print('Tru = {}, PSO = {}, Ang = {}. AS = {:.5f}. (#part = {}, #iter = {})'.format(1, 2, 'easy', as_est, npart, niter))

if plot_it:
    r = fvtk.ren()

    fvtk.add(r, fvtk.line(np.array([-sk_1[0], sk_1[0]]), fvtk.red))
    # fvtk.add(r, fvtk.line(np.array([-sk[1], sk[1]]), fvtk.red))
    # fvtk.add(r, fvtk.line(np.array([-sk[2], sk[2]]), fvtk.red))

    fvtk.add(r, fvtk.line(np.array([-sk_est[0], sk_est[0]]), fvtk.blue))
    fvtk.add(r, fvtk.line(np.array([-sk_est[1], sk_est[1]]), fvtk.blue))
    # fvtk.add(r, fvtk.line(np.array([-sk[2], sk[2]]), fvtk.blue))

    fvtk.show(r)
#######################################
pm, fV = B_N_pso(metric_for_pso_B_N_est3_1, npart, 12, niter, 0.75, 0.75, 0.75, soft_reset, good_enough, verbo)

pmm = bounds3[:, 0] + (bounds3[:, 1] - bounds3[:, 0]) * pm

_, sk_est = MultiTensor(gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]], [pmm[4], pmm[5], pmm[5]]]), angles=[(pmm[6], pmm[7]), (pmm[8], pmm[9]), (pmm[10], pmm[11])], fractions=frac3)

as_est = angular_similarity(sk_est, sk_1)
print('Tru = {}, PSO = {}, Ang = {}. AS = {:.5f}. (#part = {}, #iter = {})'.format(1, 3, 'easy', as_est, npart, niter))

if plot_it:
    r = fvtk.ren()

    fvtk.add(r, fvtk.line(np.array([-sk_1[0], sk_1[0]]), fvtk.red))
    # fvtk.add(r, fvtk.line(np.array([-sk[1], sk[1]]), fvtk.red))
    # fvtk.add(r, fvtk.line(np.array([-sk[2], sk[2]]), fvtk.red))

    fvtk.add(r, fvtk.line(np.array([-sk_est[0], sk_est[0]]), fvtk.blue))
    fvtk.add(r, fvtk.line(np.array([-sk_est[1], sk_est[1]]), fvtk.blue))
    fvtk.add(r, fvtk.line(np.array([-sk_est[2], sk_est[2]]), fvtk.blue))

    fvtk.show(r)
#######################################
#######################################
pm, fV = B_N_pso(metric_for_pso_B_N_est1_2_easy, npart, 4, niter, 0.75, 0.75, 0.75, soft_reset, good_enough, verbo)

pmm = bounds1[:, 0] + (bounds1[:, 1] - bounds1[:, 0]) * pm

_, sk_est = MultiTensor(gtab, np.array([[pmm[0], pmm[1], pmm[1]]]), angles=[(pmm[2], pmm[3])], fractions=frac1)

as_est = angular_similarity(sk_est, sk_2_easy)
print('Tru = {}, PSO = {}, Ang = {}. AS = {:.5f}. (#part = {}, #iter = {})'.format(2, 1, 'easy', as_est, npart, niter))

if plot_it:
    r = fvtk.ren()

    fvtk.add(r, fvtk.line(np.array([-sk_2_easy[0], sk_2_easy[0]]), fvtk.red))
    fvtk.add(r, fvtk.line(np.array([-sk_2_easy[1], sk_2_easy[1]]), fvtk.red))
    # fvtk.add(r, fvtk.line(np.array([-sk[2], sk[2]]), fvtk.red))

    fvtk.add(r, fvtk.line(np.array([-sk_est[0], sk_est[0]]), fvtk.blue))
    # fvtk.add(r, fvtk.line(np.array([-sk[1], sk[1]]), fvtk.blue))
    # fvtk.add(r, fvtk.line(np.array([-sk[2], sk[2]]), fvtk.blue))

    fvtk.show(r)
#######################################
pm, fV = B_N_pso(metric_for_pso_B_N_est2_2_easy, npart, 8, niter, 0.75, 0.75, 0.75, soft_reset, good_enough, verbo)

pmm = bounds2[:, 0] + (bounds2[:, 1] - bounds2[:, 0]) * pm

_, sk_est = MultiTensor(gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]]]), angles=[(pmm[4], pmm[5]), (pmm[6], pmm[7])], fractions=frac2)

as_est = angular_similarity(sk_est, sk_2_easy)
print('Tru = {}, PSO = {}, Ang = {}. AS = {:.5f}. (#part = {}, #iter = {})'.format(2, 2, 'easy', as_est, npart, niter))

if plot_it:
    r = fvtk.ren()

    fvtk.add(r, fvtk.line(np.array([-sk_2_easy[0], sk_2_easy[0]]), fvtk.red))
    fvtk.add(r, fvtk.line(np.array([-sk_2_easy[1], sk_2_easy[1]]), fvtk.red))
    # fvtk.add(r, fvtk.line(np.array([-sk[2], sk[2]]), fvtk.red))

    fvtk.add(r, fvtk.line(np.array([-sk_est[0], sk_est[0]]), fvtk.blue))
    fvtk.add(r, fvtk.line(np.array([-sk_est[1], sk_est[1]]), fvtk.blue))
    # fvtk.add(r, fvtk.line(np.array([-sk[2], sk[2]]), fvtk.blue))

    fvtk.show(r)
#######################################
pm, fV = B_N_pso(metric_for_pso_B_N_est3_2_easy, npart, 12, niter, 0.75, 0.75, 0.75, soft_reset, good_enough, verbo)

pmm = bounds3[:, 0] + (bounds3[:, 1] - bounds3[:, 0]) * pm

_, sk_est = MultiTensor(gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]], [pmm[4], pmm[5], pmm[5]]]), angles=[(pmm[6], pmm[7]), (pmm[8], pmm[9]), (pmm[10], pmm[11])], fractions=frac3)

as_est = angular_similarity(sk_est, sk_2_easy)
print('Tru = {}, PSO = {}, Ang = {}. AS = {:.5f}. (#part = {}, #iter = {})'.format(2, 3, 'easy', as_est, npart, niter))

if plot_it:
    r = fvtk.ren()

    fvtk.add(r, fvtk.line(np.array([-sk_2_easy[0], sk_2_easy[0]]), fvtk.red))
    fvtk.add(r, fvtk.line(np.array([-sk_2_easy[1], sk_2_easy[1]]), fvtk.red))
    # fvtk.add(r, fvtk.line(np.array([-sk[2], sk[2]]), fvtk.red))

    fvtk.add(r, fvtk.line(np.array([-sk_est[0], sk_est[0]]), fvtk.blue))
    fvtk.add(r, fvtk.line(np.array([-sk_est[1], sk_est[1]]), fvtk.blue))
    fvtk.add(r, fvtk.line(np.array([-sk_est[2], sk_est[2]]), fvtk.blue))

    fvtk.show(r)
#######################################
pm, fV = B_N_pso(metric_for_pso_B_N_est1_2_tight, npart, 4, niter, 0.75, 0.75, 0.75, soft_reset, good_enough, verbo)

pmm = bounds1[:, 0] + (bounds1[:, 1] - bounds1[:, 0]) * pm

_, sk_est = MultiTensor(gtab, np.array([[pmm[0], pmm[1], pmm[1]]]), angles=[(pmm[2], pmm[3])], fractions=frac1)

as_est = angular_similarity(sk_est, sk_2_tight)
print('Tru = {}, PSO = {}, Ang = {}. AS = {:.5f}. (#part = {}, #iter = {})'.format(2, 1, 'tight', as_est, npart, niter))

if plot_it:
    r = fvtk.ren()

    fvtk.add(r, fvtk.line(np.array([-sk_2_tight[0], sk_2_tight[0]]), fvtk.red))
    fvtk.add(r, fvtk.line(np.array([-sk_2_tight[1], sk_2_tight[1]]), fvtk.red))
    # fvtk.add(r, fvtk.line(np.array([-sk[2], sk[2]]), fvtk.red))

    fvtk.add(r, fvtk.line(np.array([-sk_est[0], sk_est[0]]), fvtk.blue))
    # fvtk.add(r, fvtk.line(np.array([-sk[1], sk[1]]), fvtk.blue))
    # fvtk.add(r, fvtk.line(np.array([-sk[2], sk[2]]), fvtk.blue))

    fvtk.show(r)
#######################################
pm, fV = B_N_pso(metric_for_pso_B_N_est2_2_tight, npart, 8, niter, 0.75, 0.75, 0.75, soft_reset, good_enough, verbo)

pmm = bounds2[:, 0] + (bounds2[:, 1] - bounds2[:, 0]) * pm

_, sk_est = MultiTensor(gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]]]), angles=[(pmm[4], pmm[5]), (pmm[6], pmm[7])], fractions=frac2)

as_est = angular_similarity(sk_est, sk_2_tight)
print('Tru = {}, PSO = {}, Ang = {}. AS = {:.5f}. (#part = {}, #iter = {})'.format(2, 2, 'tight', as_est, npart, niter))

if plot_it:
    r = fvtk.ren()

    fvtk.add(r, fvtk.line(np.array([-sk_2_tight[0], sk_2_tight[0]]), fvtk.red))
    fvtk.add(r, fvtk.line(np.array([-sk_2_tight[1], sk_2_tight[1]]), fvtk.red))
    # fvtk.add(r, fvtk.line(np.array([-sk[2], sk[2]]), fvtk.red))

    fvtk.add(r, fvtk.line(np.array([-sk_est[0], sk_est[0]]), fvtk.blue))
    fvtk.add(r, fvtk.line(np.array([-sk_est[1], sk_est[1]]), fvtk.blue))
    # fvtk.add(r, fvtk.line(np.array([-sk[2], sk[2]]), fvtk.blue))

    fvtk.show(r)
#######################################
pm, fV = B_N_pso(metric_for_pso_B_N_est3_2_tight, npart, 12, niter, 0.75, 0.75, 0.75, soft_reset, good_enough, verbo)

pmm = bounds3[:, 0] + (bounds3[:, 1] - bounds3[:, 0]) * pm

_, sk_est = MultiTensor(gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]], [pmm[4], pmm[5], pmm[5]]]), angles=[(pmm[6], pmm[7]), (pmm[8], pmm[9]), (pmm[10], pmm[11])], fractions=frac3)

as_est = angular_similarity(sk_est, sk_2_tight)
print('Tru = {}, PSO = {}, Ang = {}. AS = {:.5f}. (#part = {}, #iter = {})'.format(2, 3, 'tight', as_est, npart, niter))

if plot_it:
    r = fvtk.ren()

    fvtk.add(r, fvtk.line(np.array([-sk_2_tight[0], sk_2_tight[0]]), fvtk.red))
    fvtk.add(r, fvtk.line(np.array([-sk_2_tight[1], sk_2_tight[1]]), fvtk.red))
    # fvtk.add(r, fvtk.line(np.array([-sk[2], sk[2]]), fvtk.red))

    fvtk.add(r, fvtk.line(np.array([-sk_est[0], sk_est[0]]), fvtk.blue))
    fvtk.add(r, fvtk.line(np.array([-sk_est[1], sk_est[1]]), fvtk.blue))
    fvtk.add(r, fvtk.line(np.array([-sk_est[2], sk_est[2]]), fvtk.blue))

    fvtk.show(r)
#######################################
#######################################
#######################################
pm, fV = B_N_pso(metric_for_pso_B_N_est1_3_easy, npart, 4, niter, 0.75, 0.75, 0.75, soft_reset, good_enough, verbo)

pmm = bounds1[:, 0] + (bounds1[:, 1] - bounds1[:, 0]) * pm

_, sk_est = MultiTensor(gtab, np.array([[pmm[0], pmm[1], pmm[1]]]), angles=[(pmm[2], pmm[3])], fractions=frac1)

as_est = angular_similarity(sk_est, sk_3_easy)
print('Tru = {}, PSO = {}, Ang = {}. AS = {:.5f}. (#part = {}, #iter = {})'.format(3, 1, 'easy', as_est, npart, niter))

if plot_it:
    r = fvtk.ren()

    fvtk.add(r, fvtk.line(np.array([-sk_3_easy[0], sk_3_easy[0]]), fvtk.red))
    fvtk.add(r, fvtk.line(np.array([-sk_3_easy[1], sk_3_easy[1]]), fvtk.red))
    fvtk.add(r, fvtk.line(np.array([-sk_3_easy[2], sk_3_easy[2]]), fvtk.red))

    fvtk.add(r, fvtk.line(np.array([-sk_est[0], sk_est[0]]), fvtk.blue))
    # fvtk.add(r, fvtk.line(np.array([-sk[1], sk[1]]), fvtk.blue))
    # fvtk.add(r, fvtk.line(np.array([-sk[2], sk[2]]), fvtk.blue))

    fvtk.show(r)
#######################################
pm, fV = B_N_pso(metric_for_pso_B_N_est2_3_easy, npart, 8, niter, 0.75, 0.75, 0.75, soft_reset, good_enough, verbo)

pmm = bounds2[:, 0] + (bounds2[:, 1] - bounds2[:, 0]) * pm

_, sk_est = MultiTensor(gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]]]), angles=[(pmm[4], pmm[5]), (pmm[6], pmm[7])], fractions=frac2)

as_est = angular_similarity(sk_est, sk_3_easy)
print('Tru = {}, PSO = {}, Ang = {}. AS = {:.5f}. (#part = {}, #iter = {})'.format(3, 2, 'easy', as_est, npart, niter))

if plot_it:
    r = fvtk.ren()

    fvtk.add(r, fvtk.line(np.array([-sk_3_easy[0], sk_3_easy[0]]), fvtk.red))
    fvtk.add(r, fvtk.line(np.array([-sk_3_easy[1], sk_3_easy[1]]), fvtk.red))
    fvtk.add(r, fvtk.line(np.array([-sk_3_easy[2], sk_3_easy[2]]), fvtk.red))

    fvtk.add(r, fvtk.line(np.array([-sk_est[0], sk_est[0]]), fvtk.blue))
    fvtk.add(r, fvtk.line(np.array([-sk_est[1], sk_est[1]]), fvtk.blue))
    # fvtk.add(r, fvtk.line(np.array([-sk[2], sk[2]]), fvtk.blue))

    fvtk.show(r)
#######################################
pm, fV = B_N_pso(metric_for_pso_B_N_est3_3_easy, npart, 12, niter, 0.75, 0.75, 0.75, soft_reset, good_enough, verbo)

pmm = bounds3[:, 0] + (bounds3[:, 1] - bounds3[:, 0]) * pm

_, sk_est = MultiTensor(gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]], [pmm[4], pmm[5], pmm[5]]]), angles=[(pmm[6], pmm[7]), (pmm[8], pmm[9]), (pmm[10], pmm[11])], fractions=frac3)

as_est = angular_similarity(sk_est, sk_3_easy)
print('Tru = {}, PSO = {}, Ang = {}. AS = {:.5f}. (#part = {}, #iter = {})'.format(3, 3, 'easy', as_est, npart, niter))

if plot_it:
    r = fvtk.ren()

    fvtk.add(r, fvtk.line(np.array([-sk_3_easy[0], sk_3_easy[0]]), fvtk.red))
    fvtk.add(r, fvtk.line(np.array([-sk_3_easy[1], sk_3_easy[1]]), fvtk.red))
    fvtk.add(r, fvtk.line(np.array([-sk_3_easy[2], sk_3_easy[2]]), fvtk.red))

    fvtk.add(r, fvtk.line(np.array([-sk_est[0], sk_est[0]]), fvtk.blue))
    fvtk.add(r, fvtk.line(np.array([-sk_est[1], sk_est[1]]), fvtk.blue))
    fvtk.add(r, fvtk.line(np.array([-sk_est[2], sk_est[2]]), fvtk.blue))

    fvtk.show(r)
#######################################
pm, fV = B_N_pso(metric_for_pso_B_N_est1_3_tight, npart, 4, niter, 0.75, 0.75, 0.75, soft_reset, good_enough, verbo)

pmm = bounds1[:, 0] + (bounds1[:, 1] - bounds1[:, 0]) * pm

_, sk_est = MultiTensor(gtab, np.array([[pmm[0], pmm[1], pmm[1]]]), angles=[(pmm[2], pmm[3])], fractions=frac1)

as_est = angular_similarity(sk_est, sk_3_tight)
print('Tru = {}, PSO = {}, Ang = {}. AS = {:.5f}. (#part = {}, #iter = {})'.format(3, 1, 'tight', as_est, npart, niter))

if plot_it:
    r = fvtk.ren()

    fvtk.add(r, fvtk.line(np.array([-sk_3_tight[0], sk_3_tight[0]]), fvtk.red))
    fvtk.add(r, fvtk.line(np.array([-sk_3_tight[1], sk_3_tight[1]]), fvtk.red))
    fvtk.add(r, fvtk.line(np.array([-sk_3_tight[2], sk_3_tight[2]]), fvtk.red))

    fvtk.add(r, fvtk.line(np.array([-sk_est[0], sk_est[0]]), fvtk.blue))
    # fvtk.add(r, fvtk.line(np.array([-sk[1], sk[1]]), fvtk.blue))
    # fvtk.add(r, fvtk.line(np.array([-sk[2], sk[2]]), fvtk.blue))

    fvtk.show(r)
#######################################
pm, fV = B_N_pso(metric_for_pso_B_N_est2_3_tight, npart, 8, niter, 0.75, 0.75, 0.75, soft_reset, good_enough, verbo)

pmm = bounds2[:, 0] + (bounds2[:, 1] - bounds2[:, 0]) * pm

_, sk_est = MultiTensor(gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]]]), angles=[(pmm[4], pmm[5]), (pmm[6], pmm[7])], fractions=frac2)

as_est = angular_similarity(sk_est, sk_3_tight)
print('Tru = {}, PSO = {}, Ang = {}. AS = {:.5f}. (#part = {}, #iter = {})'.format(3, 2, 'tight', as_est, npart, niter))

if plot_it:
    r = fvtk.ren()

    fvtk.add(r, fvtk.line(np.array([-sk_3_tight[0], sk_3_tight[0]]), fvtk.red))
    fvtk.add(r, fvtk.line(np.array([-sk_3_tight[1], sk_3_tight[1]]), fvtk.red))
    fvtk.add(r, fvtk.line(np.array([-sk_3_tight[2], sk_3_tight[2]]), fvtk.red))

    fvtk.add(r, fvtk.line(np.array([-sk_est[0], sk_est[0]]), fvtk.blue))
    fvtk.add(r, fvtk.line(np.array([-sk_est[1], sk_est[1]]), fvtk.blue))
    # fvtk.add(r, fvtk.line(np.array([-sk[2], sk[2]]), fvtk.blue))

    fvtk.show(r)
#######################################
pm, fV = B_N_pso(metric_for_pso_B_N_est3_3_tight, npart, 12, niter, 0.75, 0.75, 0.75, soft_reset, good_enough, verbo)

pmm = bounds3[:, 0] + (bounds3[:, 1] - bounds3[:, 0]) * pm

_, sk_est = MultiTensor(gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]], [pmm[4], pmm[5], pmm[5]]]), angles=[(pmm[6], pmm[7]), (pmm[8], pmm[9]), (pmm[10], pmm[11])], fractions=frac3)

as_est = angular_similarity(sk_est, sk_3_tight)
print('Tru = {}, PSO = {}, Ang = {}. AS = {:.5f}. (#part = {}, #iter = {})'.format(3, 3, 'tight', as_est, npart, niter))

if plot_it:
    r = fvtk.ren()

    fvtk.add(r, fvtk.line(np.array([-sk_3_tight[0], sk_3_tight[0]]), fvtk.red))
    fvtk.add(r, fvtk.line(np.array([-sk_3_tight[1], sk_3_tight[1]]), fvtk.red))
    fvtk.add(r, fvtk.line(np.array([-sk_3_tight[2], sk_3_tight[2]]), fvtk.red))

    fvtk.add(r, fvtk.line(np.array([-sk_est[0], sk_est[0]]), fvtk.blue))
    fvtk.add(r, fvtk.line(np.array([-sk_est[1], sk_est[1]]), fvtk.blue))
    fvtk.add(r, fvtk.line(np.array([-sk_est[2], sk_est[2]]), fvtk.blue))

    fvtk.show(r)
#######################################
