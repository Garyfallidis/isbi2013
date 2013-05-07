from nlmeans import nlmeans
import nibabel as nib
import os

files = ['dsi_cut_v2.nii.gz']
patch_size = 1
nbhood_size = 5
sigma = 0

for img in files:

    temp, ext = str.split(os.path.basename(img), '.', 1)
    filename = os.path.dirname(os.path.realpath(img)) + '/' + temp + '_denoised.nii.gz'
        # '/data/' + temp + '_denoised_nlmeans_rician.nii.gz'

    denoised = nlmeans(nib.load(img), std=sigma, nbhood_size=patch_size, search_size=nbhood_size)  # './data/' + img))

    # Patch the b0 to 32767 like it was before denoising
    data_denoised = denoised.get_data()
    # data_denoised[...,0] = 32767
    img_denoised = nib.Nifti1Image(data_denoised, denoised.get_affine(), denoised.get_header())

    nib.save(img_denoised, filename)
