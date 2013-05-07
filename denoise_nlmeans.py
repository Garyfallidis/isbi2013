from nlmeans import nlmeans
import nibabel as nib
import os

files = ['DWIS_dsi-scheme_SNR-20.nii.gz']


for img in files:

    temp, ext =  str.split(os.path.basename(img), '.', 1)
    filename = os.path.dirname(os.path.realpath(img)) + '/data/' + temp + '_denoised_nlmeans_rician.nii.gz'
    
    denoised = nlmeans(nib.load('./data/' + img))
    
    # Patch the b0 to 32767 like it was before denoising
    data_denoised = denoised.get_data()
    data_denoised[...,0] = 32767
    img_denoised = nib.Nifti1Image(data_denoised, denoised.get_affine(), denoised.get_header())

    nib.save(img_denoised, filename)
