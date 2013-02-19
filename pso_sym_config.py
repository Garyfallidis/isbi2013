



sphere = get_sphere('symmetric724')
sphere = sphere.subdivide(1)

_, _, gtab_full = get_train_dsi(30)

gtab = deepcopy(gtab_full)

#subset of dsi gtab
bmin = 1500
bmax = 4000
gtab.b0s_mask = gtab.b0s_mask[(gtab.bvals >= bmin) & (gtab.bvals <= bmax)]
gtab.bvecs = gtab.bvecs[(gtab.bvals >= bmin) & (gtab.bvals <= bmax)]
gtab.bvals = gtab.bvals[(gtab.bvals >= bmin) & (gtab.bvals <= bmax)]
NN = gtab.bvals.shape[0]

SNR = 30.

print('SNR = {} with {} gradients direction (b{}-b{})'.format(SNR, gtab.bvals.shape[0], bmin, bmax))





#Test all cases: 1-3 trueFiber with 1-3 estFiber in pso, all with equal angles (not in pso) and equal eVals, for easy and tight angles

mevals1 = np.array([0.0017, 0.0003, 0.0003])
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
    return (wts * ((S - S_gt) ** 2)).sum()



bounds1 = np.array([[0.001, 0.003], [0.0001, 0.0005], [0, 180], [-90, 90]])  # ,[25, 75]])
bounds2 = np.array([[0.001, 0.003], [0.0001, 0.0005], [0.001, 0.003], [0.0001, 0.0005], [0, 180], [-90, 90], [0, 180], [-90, 90]])  # ,[25, 75]])
bounds3 = np.array([[0.001, 0.003], [0.0001, 0.0005], [0.001, 0.003], [0.0001, 0.0005], [0.001, 0.003], [0.0001, 0.0005], [0, 180], [-90, 90], [0, 180], [-90, 90], [0, 180], [-90, 90]])  # ,[25, 75]])


def metric_for_pso_B_N_1(pm):
    pmm = bounds[:, 0] + (bounds1[:, 1] - bounds1[:, 0]) * pm
    return fit_quality_mt(S_1, gtab, np.array([[pmm[0], pmm[1], pmm[1]]]), angles=[(pmm[2], pmm[3])], fractions=frac1)

def metric_for_pso_B_N_2_easy(pm):
    pmm = bounds[:, 0] + (bounds2[:, 1] - bounds2[:, 0]) * pm
    return fit_quality_mt(S_2_easy, gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]]]), angles=[(pmm[4], pmm[5]), (pmm[6], pmm[7])], fractions=frac2)

def metric_for_pso_B_N_2_tight(pm):
    pmm = bounds[:, 0] + (bounds2[:, 1] - bounds2[:, 0]) * pm
    return fit_quality_mt(S_2_tight, gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]]]), angles=[(pmm[4], pmm[5]), (pmm[6], pmm[7])], fractions=frac2)

def metric_for_pso_B_N_3_easy(pm):
    pmm = bounds[:, 0] + (bounds3[:, 1] - bounds3[:, 0]) * pm
    return fit_quality_mt(S_3_easy, gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]], [pmm[4], pmm[5], pmm[5]]]), angles=[(pmm[6], pmm[7]), (pmm[8], pmm[9]), (pmm[10], pmm[11])], fractions=frac3)

def metric_for_pso_B_N_3_tight(pm):
    pmm = bounds[:, 0] + (bounds3[:, 1] - bounds3[:, 0]) * pm
    return fit_quality_mt(S_3_tight, gtab, np.array([[pmm[0], pmm[1], pmm[1]], [pmm[2], pmm[3], pmm[3]], [pmm[4], pmm[5], pmm[5]]]), angles=[(pmm[6], pmm[7]), (pmm[8], pmm[9]), (pmm[10], pmm[11])], fractions=frac3)

