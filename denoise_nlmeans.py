from nlmeans import nlmeans
import nibabel as nib
import os

files = ['training-data_DWIS_dsi-scheme_SNR-10.nii.gz',
         'training-data_DWIS_dsi-scheme_SNR-30.nii.gz',
         'training-data_DWIS_dti-scheme_SNR-10.nii.gz',
         'training-data_DWIS_dti-scheme_SNR-30.nii.gz',
         'training-data_DWIS_hardi-scheme_SNR-10.nii.gz',
         'training-data_DWIS_hardi-scheme_SNR-10.nii.gz']


for img in files:

    temp, ext =  str.split(os.path.basename(img), '.', 1)
    filename = os.path.dirname(os.path.realpath(img)) + '/data/' + temp + '_denoised_nlmeans.nii.gz'
    nib.save(nlmeans(nib.load('./data/' + img)), filename)