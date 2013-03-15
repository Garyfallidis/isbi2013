import numpy as np
import nibabel as nib
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table

"""
0 : no denoising
1 : denoising using AONLM
2 : NLMeans with rician averaging denoising
3 : NLMeans with gaussian averaging denoising
"""

dname = 'data/'


def read_data(fimg, fbvals, fbvecs):
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs)
    img = nib.load(fimg)
    data = img.get_data()
    affine = img.get_affine()
    return data, affine, gtab


def get_train_dti(snr=30, denoised=0):
    if denoised == 0:
        den = ''
        dname2 = ''
        
    elif denoised == 1:
        den = '_AONLM_1.00_rician'
        dname2 = 'training_denoised/Coupe/'

    elif denoised == 2:
        den = '_denoised_nlmeans_rician'
        dname2 = 'training_denoised/NLM_Rician/'

    elif denoised == 3:
        den = '_denoised_nlmeans'
        dname2 = 'training_denoised/NLM_Gaussian/'

    fimg = dname + dname2 + 'training-data_DWIS_dti-scheme_SNR-' + str(snr) + den + '.nii.gz'
    fbvals = dname + 'dti-scheme.bval'
    fbvecs = dname + 'dti-scheme.bvec'
    return read_data(fimg, fbvals, fbvecs)


def get_test_dti(snr=30, denoised=0):
    if denoised == 0:
        den = ''
        dname2 = 'elef_testing/'
        
    elif denoised == 1:
        den = '_AONLM_1.00_rician'
        dname2 = 'elef_testing/Coupe/'

    elif denoised == 2:
        den = '_denoised_nlmeans_rician'
        dname2 = 'elef_testing/NLM_Rician/'

    elif denoised == 3:
        den = '_denoised_nlmeans'
        dname2 = 'elef_testing/NLM_Gaussian/'

    fimg = dname + dname2 + 'DWIS_dti-scheme_SNR-' + str(snr) + den + '.nii.gz'
    fbvals = dname + 'dti-scheme.bval'
    fbvecs = dname + 'dti-scheme.bvec'
    return read_data(fimg, fbvals, fbvecs)


def get_train_hardi(snr=30, denoised=0):
    if denoised == 0:
        den = ''
        dname2 = ''
        
    elif denoised == 1:
        den = '_AONLM_1.00_rician'
        dname2 = 'training_denoised/Coupe/'

    elif denoised == 2:
        den = '_denoised_nlmeans_rician'
        dname2 = 'training_denoised/NLM_Rician/'

    elif denoised == 3:
        den = '_denoised_nlmeans'
        dname2 = 'training_denoised/NLM_Gaussian/'

    fimg = dname + dname2 + 'training-data_DWIS_hardi-scheme_SNR-' + str(snr) + den + '.nii.gz'
    fbvals = dname + 'hardi-scheme.bval'
    fbvecs = dname + 'hardi-scheme.bvec'
    return read_data(fimg, fbvals, fbvecs)


def get_test_hardi(snr=30, denoised=0):
    if denoised == 0:
        den = ''
        dname2 = 'elef_testing/'
        
    elif denoised == 1:
        den = '_AONLM_1.00_rician'
        dname2 = 'elef_testing/Coupe/'

    elif denoised == 2:
        den = '_denoised_nlmeans_rician'
        dname2 = 'elef_testing/NLM_Rician/'

    elif denoised == 3:
        den = '_denoised_nlmeans'
        dname2 = 'elef_testing/NLM_Gaussian/'

    fimg = dname + dname2 + 'DWIS_hardi-scheme_SNR-' + str(snr) + den + '.nii.gz'
    fbvals = dname + 'hardi-scheme.bval'
    fbvecs = dname + 'hardi-scheme.bvec'
    return read_data(fimg, fbvals, fbvecs)


def get_train_dsi(snr=30, denoised=0):
    if denoised == 0:
        den = ''
        dname2 = ''
        
    elif denoised == 1:
        den = '_AONLM_1.00_gauss'
        dname2 = 'training_denoised/Coupe/'

    elif denoised == 2:
        den = '_denoised_nlmeans_rician'
        dname2 = 'training_denoised/NLM_Rician/'

    elif denoised == 3:
        den = '_denoised_nlmeans'
        dname2 = 'training_denoised/NLM_Gaussian/'

    fimg = dname + dname2 + 'training-data_DWIS_dsi-scheme_SNR-' + str(snr) + den + '.nii.gz'
    fbvals = dname + 'dsi-scheme.bval'
    fbvecs = dname + 'dsi-scheme.bvec'
    return read_data(fimg, fbvals, fbvecs)


def get_test_dsi(snr=30, denoised=0):
    if denoised == 0:
        den = ''
        dname2 = 'elef_testing/'
        
    elif denoised == 1:
        den = '_AONLM_1.00_gauss'
        dname2 = 'elef_testing/Coupe/'

    elif denoised == 2:
        den = '_denoised_nlmeans_rician'
        dname2 = 'elef_testing/NLM_Rician/'

    elif denoised == 3:
        den = '_denoised_nlmeans'
        dname2 = 'elef_testing/NLM_Gaussian/'

    fimg = dname + dname2 + 'DWIS_dsi-scheme_SNR-' + str(snr) + den + '.nii.gz'
    fbvals = dname + 'dsi-scheme.bval'
    fbvecs = dname + 'dsi-scheme.bvec'
    return read_data(fimg, fbvals, fbvecs)


def get_specific_data(training, category, snr, denoise):

    if denoise == False:
        denoise = 0

    if training == True:
        if category == 'dti':
            return get_train_dti(snr, denoise)
        if category == 'hardi':
            return get_train_hardi(snr, denoise)
        if category == 'dsi':
            return get_train_dsi(snr, denoise)

    if training == False:
        if category == 'dti':
            return get_test_dti(snr, denoise)
        if category == 'hardi':
            return get_test_hardi(snr, denoise)
        if category == 'dsi':
            return get_test_dsi(snr, denoise)



def get_train_mask():
    fimg = dname + 'training-data_mask.nii.gz'
    img = nib.load(fimg)
    return img.get_data(), img.get_affine()


def get_train_rois():
    fimg = dname + 'training-data_rois.nii.gz'
    img = nib.load(fimg)
    return img.get_data(), img.get_affine()


def get_train_gt_fibers():
    streamlines = []

    for i in range(1, 21):
        ffib = dname + '/ground-truth-fibers/fiber-'
        ffib += str(i).zfill(2) + '.txt'
        streamlines.append(np.loadtxt(ffib))

    fradii = dname + '/ground-truth-fibers/fibers-radii.txt'

    return streamlines, np.loadtxt(fradii)
