import numpy as np
# from copy import deepcopy

#dont use that
def basic_pso(f, S, n, bounds, maxit, w, phi_p, phi_g, verbose):
    # basic particle swarm optimization (PSO)
    # f is the function to minimize, f:R^n -> R
    # S is the number of particle in the swarm
    # n is the dimension of the search space
    # bounds is the lower and higher bounds on each dimension, shape = (n,2)
    # maxit is the maximum number of iteration
    # w, phi_p and phi_g are user specified paramters controlling the behavior and efficacity of PSO
    # verbose print iteration progress if == 1

    # initialise S particle in Uniform of bounds
    space_width = bounds[:, 1] - bounds[:, 0]
    swarm = np.tile(bounds[:, 0], (S, 1)) + np.random.rand(S, n) * np.tile(space_width, (S, 1))

    # particles best known position
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

    # initialise particles velocity
    v = np.tile(-space_width, (S, 1)) + np.random.rand(S, n) * np.tile(2 * space_width, (S, 1))

    iter = 0
    while iter < maxit:
        # update velocity
        v = w * v + phi_p * np.random.rand(S, n) * (p - swarm) + phi_g * np.random.rand(S, n) * (np.tile(g, (S, 1)) - swarm)
        # update swarm
        swarm += v

        # update particle's position value and best position
        for part in range(S):
            fswarm[part] = f(swarm[part, :].squeeze())
            if fswarm[part] < fp[part]:
                fp[part] = fswarm[part]
                p[part, :] = swarm[part, :]

        # update swarm best known position
        if np.min(fp) < fg:
            fg = np.min(fp)
            g = p[np.argmin(fp), :]

        if verbose:
            if (iter) % 1 == 0:
                print('iter = {}, best value = {}'.format(iter, fg))

        iter += 1

    return g, fg


def B_N_pso(f, S, n, maxit, w, phi_p, phi_g, soft_reset, good_enough, verbose):
    # basic particle swarm optimization (PSO)
    # f is the function to minimize, f:R^n -> R
    # S is the number of particle in the swarm
    # n is the dimension of the search space
    # bounds is the lower and higher bounds on each dimension, shape = (n,2)
    # maxit is the maximum number of iteration
    # w, phi_p and phi_g are user specified paramters controlling the behavior and efficacity of PSO
    # verbose print iteration progress if == 1

    # initialise S particle in Uniform random of [0,1]
    swarm = np.random.rand(S, n)

    # particles best known position
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

    # initialise particles velocity
    v = (np.tile(-1, (S, 1)) + np.random.rand(S, n) * np.tile(2, (S, 1)))

    all_best = np.zeros(maxit + 1)
    all_best[0] = fg
    iter = 0
    # number of iteration before we activated soft_reset
    flag = 75
    while (iter < maxit) and (fg > good_enough):
        # update velocity
        v = w * v + phi_p * np.random.rand(S, n) * (p - swarm) + phi_g * np.random.rand(S, n) * (np.tile(g, (S, 1)) - swarm)
        # update swarm
        swarm += v

        #bound the parameter space and kill out off bound velocity
        v[swarm < 0] = 0
        swarm[swarm < 0] = 0

        v[swarm > 1] = 0
        swarm[swarm > 1] = 1

        # update particle's position value and best position
        for part in range(S):
            fswarm[part] = f(swarm[part, :].squeeze())
            if fswarm[part] < fp[part]:
                fp[part] = fswarm[part]
                p[part, :] = swarm[part, :]

        # update swarm best known position
        if np.min(fp) < fg:
            fg = np.min(fp)
            g = p[np.argmin(fp), :]

        all_best[iter + 1] = fg

        # if soft_reset == 1:
        #         #random reset every 25 iter
        #     if ((iter - 24) % 25) == 0:
        #         swarm = np.random.rand(S, n)
        #         v = np.zeros((S, n))

        #     if ((iter - 49) % 50) == 0:
        #         p = swarm.copy()
        #         fp = np.inf * np.ones(S)

        if soft_reset == 1:
            if flag <= 0:
                if (all_best[iter - 25] - all_best[iter]) < 0.025 * all_best[iter - 25]:
                    #randomly move particule
                    swarm = np.random.rand(S, n)
                    v = np.zeros((S, n))

                    p = swarm.copy()
                    fp = np.inf * np.ones(S)
                    
                    #number of iteration after a soft_reset before we can reset again
                    flag = 25
                    print('iter = {}, best value = {}, RESET!'.format(iter, fg))

        if verbose:
            if (iter % 25) == 0:
                print('iter = {}, best value = {}'.format(iter, fg))

        iter += 1
        flag -= 1
    print('iter = {}, best value = {}'.format(iter, fg))
    return g, fg

#this is crap
def B_N_H_pso(f, S, n, maxit, w, phi_p, phi_g, verbose):
    # basic particle swarm optimization (PSO)
    # f is the function to minimize, f:R^n -> R
    # S is the number of particle in the swarm
    # n is the dimension of the search space
    # bounds is the lower and higher bounds on each dimension, shape = (n,2)
    # maxit is the maximum number of iteration
    # w, phi_p and phi_g are user specified paramters controlling the behavior and efficacity of PSO
    # verbose print iteration progress if == 1

    # np.random.seed(1)
    init_boost = 10
    # initialise 10*S particle in Uniform random of [0,1]
    # keep S best
    swarm = np.random.rand(init_boost * S, n)

    # particles best known position
    p = swarm.copy()

    # evaluate particle's best position value
    fp = np.zeros(init_boost * S)
    for part in range(init_boost * S):
        fp[part] = f(p[part, :].squeeze())

    # prune init particules
    idx = np.argsort(fp)
    swarm = swarm[idx[:S], :]
    p = swarm.copy()
    fp = fp[idx[:S]]

    # evaluate particle's position value
    fswarm = fp.copy()

    # swarm's best known position
    # g = p[np.argmin(fp),:]
    g = p[0, :]
    # swarm best known value
    # fg = np.min(fp)
    fg = fp[0]

    # initialise particles velocity
    v = (np.tile(-1, (S, 1)) + np.random.rand(S, n) * np.tile(2, (S, 1)))

    iter = 0
    while iter < maxit:
        # update velocity
        v = w * v + phi_p * np.random.rand(S, n) * (p - swarm) + phi_g * np.random.rand(S, n) * (np.tile(g, (S, 1)) - swarm)
        # update swarm
        swarm += v

        #bound the parameter space and kill out off bound velocity
        v[swarm < 0] = 0
        swarm[swarm < 0] = 0

        v[swarm > 1] = 0
        swarm[swarm > 1] = 1

        # update particle's position value and best position
        for part in range(S):
            fswarm[part] = f(swarm[part, :].squeeze())
            if fswarm[part] < fp[part]:
                fp[part] = fswarm[part]
                p[part, :] = swarm[part, :]

        # update swarm best known position
        if np.min(fp) < fg:
            fg = np.min(fp)
            g = p[np.argmin(fp), :]

        if verbose:
            if (iter % 5) == 0:
                print('iter = {}, best value = {}'.format(iter, fg))

        iter += 1
    print('iter = {}, best value = {}'.format(iter, fg))
    return g, fg

# this is crap**2
def neigh_pso(f, S, n, bounds, maxit, w, phi_p, phi_g, m, verbose):
    # particle swarm optimization (PSO) using neighborhood
    # f is the function to minimize, f:R^n -> R
    # S is the number of particle in the swarm
    # n is the dimension of the search space
    # bounds is the lower and higher bounds on each dimension, shape = (n,2)
    # maxit is the maximum number of iteration
    # w, phi_p and phi_g are user specified paramters controlling the behavior and efficacity of PSO
    # m is the number of neighboor to include in "swarm's best"
    # verbose print iteration progress if == 1

    # initialise S particle in Uniform of bounds
    space_width = bounds[:, 1] - bounds[:, 0]
    swarm = np.tile(bounds[:, 0], (S, 1)) + np.random.rand(S, n) * np.tile(space_width, (S, 1))

    # particles best known position
    p = deepcopy(swarm)

    # evaluate particle's best position value
    fp = np.zeros(S)
    for part in range(S):
        fp[part] = f(p[part, :].squeeze())

    # evaluate particle's position value
    fswarm = deepcopy(fp)

    # swarm's best known position
    g = p[np.argmin(fp), :]
    # swarm best known value
    fg = np.min(fp)

    # initialise particles velocity
    v = np.tile(-space_width, (S, 1)) + np.random.rand(S, n) * np.tile(2 * space_width, (S, 1))

    iter = 0
    while iter < maxit:
        # update velocity
        for part in range(S):
            # find  part's m closest neighbor
            neigh_dist = ((swarm - np.tile(swarm[part, :], (S, 1))) ** 2).sum(1)
            neigh_dist[part] = np.inf
            neigh = np.argsort(neigh_dist)
            neigh = neigh[:m]
            # find best know position amongst neighbor
            g_loc = p[neigh[np.argmin(fp[neigh])], :]
            # update velocity with neighborhood
            v[part, :] = w * v[part, :] + phi_p * np.random.rand(1, n) * (p[part, :] - swarm[part, :]) + phi_g * np.random.rand(1, n) * (g_loc - swarm[part, :])

        # update swarm
        swarm += v

        # update particle's position value and best position
        for part in range(S):
            fswarm[part] = f(swarm[part, :].squeeze())
            if fswarm[part] < fp[part]:
                fp[part] = fswarm[part]
                p[part, :] = swarm[part, :]

        # update swarm best known position
        if np.min(fp) < fg:
            fg = np.min(fp)
            g = p[np.argmin(fp), :]

        if verbose:
            if (iter) % 25 == 0:
                print('iter = {}, best value = {}'.format(iter, fg))

        iter += 1

    return g, fg
