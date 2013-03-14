import numpy as np
from cleaned_pso import *
# from dipy.sims.voxel import MultiTensor
# from dipy.core.sphere_stats import angular_similarity
from load_data import get_train_dti
import nibabel as nib


def run_PSO_testingData_slice(NC=2, iso=0, fr=0, Npart=100, Niter=50, slicez=0, snr=30, deno=None, verb=0):

    if deno == 0:
        den = None
    else:
        den = 1

    data, affine, gtab = get_train_dti(snr=snr, denoised=den)
    data = data[:, :, slicez].squeeze()
    mask = nib.load('wm_mask_hardi_01.nii.gz').get_data()
    mask = mask[:, :, slicez].squeeze()

    print('{} {} {} {} {} {} {}'.format(NC, iso, fr, Npart, Niter, snr, deno))

    for ix in range(data.shape[0]):
        for iy in range(data.shape[1]):
            if mask[ix, iy]:

                sig = data[ix, iy]

                plain = run_PSO(sig, gtab, NC=NC, iso=iso, fr=fr, bounds=None, metric=0, S=Npart,
                                maxit=Niter, w=0.75, phi_p=0.5, phi_g=0.5, verbose=verb, plotting=0, out=0)

                printy = str(ix)
                printy += ' '
                printy += str(iy)
                printy += ' '
                printy += str(slicez)
                for ii in range(plain.shape[0]):
                    printy += ' '
                    printy += str(plain[ii])

                print(printy)

    # S, sticks = MultiTensor(gtab, mevals, 100, angles, fractions, None)
    # as_est = angular_similarity(sticks, d)
    # # as_iner = angular_similarity(sticks[0], sticks[1])
    # print iso, fr, as_est  # , as_iner

if __name__ == "__main__":
    import sys
    run_PSO_testingData_slice(*np.array(sys.argv[1:], np.int))
