import numpy as np
from dipy.sims.voxel import MultiTensor


def PSO_N(f, S=100, n=10, maxit=100, w=0.75, phi_p=0.5, phi_g=0.5, verbose=0):
    # basic particle swarm optimization (PSO) with normalized parameter
    #   space [0,1]
    # f is the function to minimize, f:R^n -> R
    # S is the number of particle in the swarm
    # n is the dimension of the search space
    # bounds is the lower and higher bounds on each dimension, shape = (n,2)
    # maxit is the maximum number of iteration
    # w, phi_p and phi_g are user specified parameters controlling the behavior
    #   and efficacity of PSO
    # verbose print iteration progress if == 1

    # initialise S particle in Uniform random of [0,1]
    swarm = np.random.rand(S, n)

    # particles best known position(to date)
    p = swarm.copy()

    # evaluate particle's best position value
    fp = np.zeros(S)
    for part in range(S):
        fp[part] = f(p[part, :].squeeze())

    # evaluate particle's position value
    fswarm = fp.copy()

    # swarm's best known position
    g = p[np.argmin(fp), :]
    # swarm best known value
    fg = np.min(fp)

    # initialise particles velocity in [-1,1]
    v = -1 + np.random.rand(S, n) * 2

    iter = 0
    #main loop
    while (iter < maxit):
        # update velocity
        v = w * v + phi_p * np.random.rand(S, n) * (p - swarm) + phi_g * \
            np.random.rand(S, n) * (np.tile(g, (S, 1)) - swarm)
        # update swarm
        swarm += v

        #bound the parameter space and kill out-off-bound velocities
        v[swarm < 0] = 0
        swarm[swarm < 0] = 0

        v[swarm > 1] = 0
        swarm[swarm > 1] = 1

        # update particle's position value and best position
        for part in xrange(S):
            fswarm[part] = f(swarm[part, :].squeeze())
            if fswarm[part] < fp[part]:
                fp[part] = fswarm[part]
                p[part, :] = swarm[part, :]

        # update swarm best known position
        if np.min(fp) < fg:
            fg = np.min(fp)
            g = p[np.argmin(fp), :]

        if verbose:
            if (iter % 25) == 0:
                print('iter = {}, best value = {}'.format(iter, fg))

        iter += 1
    if verbose:
        print('iter = {}, best value = {}'.format(iter, fg))
    return g, fg


def run_PSO(signal, gtab, NC=2, iso=0, fr=0, bounds=None, metric=0, S=100,
            maxit=100, w=0.75, phi_p=0.5, phi_g=0.5, verbose=0, plotting=0, out=0):

    if bounds == None:
        bounds_NC = np.array([[0.001, 0.003], [0.0001, 0.0005], [0, 180],
                              [-90, 90]])
        bounds_iso = np.array([[0.0001, 0.003], [0.0, 100.0]])
        bounds_fr = np.array([[0.0, 100.0]])

        bounds = bounds_NC
        for i in xrange(NC - 1):
            bounds = np.vstack((bounds, bounds_NC))
        if iso:
            bounds = np.vstack((bounds, bounds_iso))
        if fr:
            for i in xrange(NC - 1):
                bounds = np.vstack((bounds, bounds_fr))

    # print bounds

    def convert_params(N_params):
        params = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * N_params
        return params

    def parse_params(params):
        mevals = []
        angles = []
        fractions = []
        for i in range(NC):
            mevals.append(np.array([params[4 * i], params[4 * i + 1], params[4 * i + 1]]))
            angles.append(np.array([params[4 * i + 2], params[4 * i + 3]]))
        if iso:
            mevals.append(np.array([params[4 * NC], params[4 * NC], params[4 * NC]]))
            angles.append(np.array([0., 0.]))
            iso_frac = params[4 * NC + 1]
        if ~fr:
            if ~iso:
                fractions.append(np.ones((1, NC)) * (100. / NC))
            else:
                fractions.append(np.vstack((np.ones((NC, 1)) * (1. / NC) * (100. - iso_frac), np.array([iso_frac]))).T)
        else:
            if NC == 1:
                frac = 100.
            else:
                fracc = params[-(NC - 1)]
                frac = np.zeros(NC)
                frac[0] = fracc[0]
                if NC == 2:
                    frac[1] = 100. - frac[0]
                else:
                    frac[1] = (100. - frac[0]) * fracc[1] / 100.
                if NC == 3:
                    frac[2] = 100. - (frac[0] + frac[1])
            if ~iso:
                fractions.append(frac)
            else:
                fractions.append(np.vstack((frac * (100. - iso_frac) / 100., np.array([iso_frac]))).T)

        return np.array(mevals), np.array(angles), np.array(fractions).T

    def fit_quality(N_params):
        # print N_params
        params = convert_params(N_params)
        # print params
        mevals, angles, fractions = parse_params(params)
        # print mevals, angles, fractions
        S, _ = MultiTensor(gtab, mevals, 100, angles, fractions, None)

        if metric == 0:
            fitness_value = ((S - signal) ** 2).sum()
        return fitness_value

    N_params, fV = PSO_N(fit_quality, S=S, n=(4 + fr) * NC + 2 * iso - fr,
                         maxit=maxit, w=w, phi_p=phi_p, phi_g=phi_g, verbose=verbose)
    if out==0:
        return convert_params(N_params)
    else:
        return parse_params(convert_params(N_params))





# _, sk_est = MultiTensor(gtab, np.array([[pmm[0], pmm[1], pmm[1]],
#             [pmm[2], pmm[3], pmm[3]], [pmm[4], pmm[5], pmm[5]]]),
#             angles=[(pmm[6], pmm[7]), (pmm[8], pmm[9]), (pmm[10], pmm[11])],
#             fractions=frac3)

# as_est = angular_similarity(sk_est, sk_3_easy)
# print('Tru = {}, PSO = {}, Ang = {}. AS = {:.5f}. (#part = {}, #iter = {})'\
#         .format(3, 3, 'easy', as_est, npart, niter))

# if plot_it:
#     from dipy.viz import fvtk

# if plot_it:
#     r = fvtk.ren()

#     fvtk.add(r, fvtk.line(np.array([-sk_3_easy[0], sk_3_easy[0]]), fvtk.red))
#     fvtk.add(r, fvtk.line(np.array([-sk_3_easy[1], sk_3_easy[1]]), fvtk.red))
#     fvtk.add(r, fvtk.line(np.array([-sk_3_easy[2], sk_3_easy[2]]), fvtk.red))

#     fvtk.add(r, fvtk.line(np.array([-sk_est[0], sk_est[0]]), fvtk.blue))
#     fvtk.add(r, fvtk.line(np.array([-sk_est[1], sk_est[1]]), fvtk.blue))
#     fvtk.add(r, fvtk.line(np.array([-sk_est[2], sk_est[2]]), fvtk.blue))

#     fvtk.show(r)
