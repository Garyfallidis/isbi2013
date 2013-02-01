import nibabel as nib
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table


dname = 'data/'


def read_data(fimg, fbvals, fbvecs):
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs)
    img = nib.load(fimg)
    data = img.get_data()
    affine = img.get_affine()
    return data, affine, gtab


def get_train_dti(snr=30):
    fimg = dname + 'DWIS_dsi-scheme_SNR-' + str(snr) + '.nii.gz'
    fbvals = dname + 'dsi-scheme.bval'
    fbvecs = dname + 'dsi-scheme.bvec'
    return read_data(fimg, fbvals, fbvecs)


def get_train_hardi(snr=30):
    fimg = dname + 'DWIS_hardi-scheme_SNR-' + str(snr) + '.nii.gz'
    fbvals = dname + 'hardi-scheme.bval'
    fbvecs = dname + 'hardi-scheme.bvec'
    return read_data(fimg, fbvals, fbvecs)


def get_train_dsi(snr=30):
    fimg = dname + 'DWIS_dsi-scheme_SNR-' + str(snr) + '.nii.gz'
    fbvals = dname + 'dsi-scheme.bval'
    fbvecs = dname + 'dsi-scheme.bvec'
    return read_data(fimg, fbvals, fbvecs)


