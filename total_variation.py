import numpy as np
from tempfile import mkstemp
from os.path import join as pjoin


def tv_denoise_4d(im, weight=100, eps=2.e-4, n_iter_max=200):
    """
    Perform total-variation denoising on 4-D arrays

    Parameters
    ----------
    im: ndarray
        4-D input data to be denoised

    weight: float, optional
        denoising weight. The greater ``weight``, the more denoising (at
        the expense of fidelity to ``input``)

    eps: float, optional
        relative difference of the value of the cost function that determines
        the stop criterion. The algorithm stops when:

            (E_(n-1) - E_n) < eps * E_0

    n_iter_max: int, optional
        maximal number of iterations used for the optimization.

    Returns
    -------
    out: ndarray
        denoised array

    Notes
    -----
    Rudin, Osher and Fatemi algorithm

    Examples
    ---------
    x, y, z, k = np.ogrid[0:40, 0:40, 0:40, 0:40]
    mask = (x -22)**2 + (y - 20)**2 + (z - 17)**2 + (k -20)**2 < 8**2
    mask = mask.astype(np.float)
    mask += 0.2*np.random.randn(*mask.shape)
    res = tv_denoise_4d(mask, weight=0.1)

    """

    px = memmap_zeros(im)
    py = memmap_zeros(im)
    pz = memmap_zeros(im)
    pk = memmap_zeros(im)

    gx = memmap_zeros(im)
    gy = memmap_zeros(im)
    gz = memmap_zeros(im)
    gk = memmap_zeros(im)

    d = memmap_zeros(im)
    norm = memmap_zeros(im)

    i = 0
    while i < n_iter_max:
        d = - px - py - pz - pk
        d[1:] += px[:-1]
        d[:, 1:] += py[:, :-1]
        d[:, :, 1:] += pz[:, :, :-1]
        d[:, :, :, 1:] += pk[:, :, :, :-1]

        out = im + d
        E = (d ** 2).sum()

        gx[:-1] = np.diff(out, axis=0)
        gy[:, :-1] = np.diff(out, axis=1)
        gz[:, :, :-1] = np.diff(out, axis=2)
        gk[:, :, :, :-1] = np.diff(out, axis=3)

        norm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + gk ** 2)
        E += weight * norm.sum()
        norm *= 0.5 / weight
        norm += 1.
        px -= 1. / 8. * gx
        px /= norm
        py -= 1. / 8. * gy
        py /= norm
        pz -= 1 / 8. * gz
        pz /= norm
        pk -= 1. / 8. * gk
        pk /= norm

        E /= float(im.size)
        if i == 0:
            E_init = E
            E_previous = E
        else:
            if np.abs(E_previous - E) < eps * E_init:
                break
            else:
                E_previous = E
        i += 1

    del px, py, pz, pk, gx, gy, gz, gk, d, norm
    return out.astype('f8')


def memmap_zeros(im):
    _, filename = mkstemp()
    return np.memmap(filename, dtype='float32', mode='w+', shape=im.shape)


if __name__ == '__main__':
    """
    #test 3D
    x, y, z = np.ogrid[0:40, 0:40, 0:40]
    mask = (x -22)**2 + (y - 20)**2 + (z - 17)**2 < 8**2
    mask = mask.astype(np.float)
    mask += 0.2*np.random.randn(*mask.shape)
    res = tv_denoise_3d(mask, weight=0.1)
    #test 4D
    x, y, z, k = np.ogrid[0:40, 0:40, 0:40, 0:40]
    mask = (x -22)**2 + (y - 20)**2 + (z - 17)**2 + (k -20)**2 < 8**2
    mask = mask.astype(np.float)
    mask += 0.2*np.random.randn(*mask.shape)
    res = tv_denoise_4d(mask, weight=0.1)
    """
    import sys

    import nibabel as nib
    img = nib.load(sys.argv[1])

    data = img.get_data()
    affine = img.get_affine()

    from time import time

    t0 = time()

    data2 = tv_denoise_4d(data, np.float(sys.argv[2]))

    t1 = time()

    print 'sec', t1 - t0

    nib.save(nib.Nifti1Image(data2, affine), 'tmp_0.5.nii.gz')
