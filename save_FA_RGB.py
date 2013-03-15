import nibabel as nib
import load_data
from make_FA_RGB import FA_RGB


snr_list = [10, 30]
denoised_list = [0, 1 , 2, 3]


# Training data
for snr, denoised in [(snr, denoised) for snr in snr_list for denoised in denoised_list]:

    print "Current file : dti snr = " +  str(snr) + ' denoising = ' + str(denoised)
    data, affine, gtab = load_data.get_train_dti(snr=snr, denoised=denoised)
    FA, RGB = FA_RGB(data, gtab)
    nib.save(nib.Nifti1Image(FA, affine),  'training-data_DWIS_dti-scheme_SNR-' + str(snr) + '_denoising_' + str(denoised) + '_FA.nii.gz')
    nib.save(nib.Nifti1Image(RGB, affine), 'training-data_DWIS_dti-scheme_SNR-' + str(snr) + '_denoising_' + str(denoised) + '_RGB.nii.gz')
    

    print "Current file : dsi snr = " +  str(snr) + ' denoising = ' + str(denoised)
    data, affine, gtab = load_data.get_train_dsi(snr=snr, denoised=denoised)
    FA, RGB = FA_RGB(data, gtab)
    nib.save(nib.Nifti1Image(FA, affine),  'training-data_DWIS_dsi-scheme_SNR-' + str(snr) + '_denoising_' + str(denoised) + '_FA.nii.gz')
    nib.save(nib.Nifti1Image(RGB, affine), 'training-data_DWIS_dsi-scheme_SNR-' + str(snr) + '_denoising_' + str(denoised) + '_RGB.nii.gz')


    print "Current file : hardi snr = " +  str(snr) + ' denoising = ' + str(denoised)
    data, affine, gtab = load_data.get_train_hardi(snr=snr, denoised=denoised)
    FA, RGB = FA_RGB(data, gtab)
    nib.save(nib.Nifti1Image(FA, affine),  'training-data_DWIS_hardi-scheme_SNR-' + str(snr) + '_denoising_' + str(denoised) + '_FA.nii.gz')
    nib.save(nib.Nifti1Image(RGB, affine), 'training-data_DWIS_hardi-scheme_SNR-' + str(snr) + '_denoising_' + str(denoised) + '_RGB.nii.gz')


    

snr_list = [10, 20, 30]

# Testing data
for snr, denoised in [(snr, denoised) for snr in snr_list for denoised in denoised_list]:

    print "Current file : dti snr = " +  str(snr) + ' denoising = ' + str(denoised)
    data, affine, gtab = load_data.get_test_dti(snr=snr, denoised=denoised)
    FA, RGB = FA_RGB(data, gtab)
    nib.save(nib.Nifti1Image(FA, affine),  'DWIS_dti-scheme_SNR-' + str(snr) + '_denoising_' + str(denoised) + '_FA.nii.gz')
    nib.save(nib.Nifti1Image(RGB, affine), 'DWIS_dti-scheme_SNR-' + str(snr) + '_denoising_' + str(denoised) + '_RGB.nii.gz')


    print "Current file : dsi snr = " +  str(snr) + ' denoising = ' + str(denoised)
    data, affine, gtab = load_data.get_test_dsi(snr=snr, denoised=denoised)
    FA, RGB = FA_RGB(data, gtab)
    nib.save(nib.Nifti1Image(FA, affine),  'DWIS_dsi-scheme_SNR-' + str(snr) + '_denoising_' + str(denoised) + '_FA.nii.gz')
    nib.save(nib.Nifti1Image(RGB, affine), 'DWIS_dsi-scheme_SNR-' + str(snr) + '_denoising_' + str(denoised) + '_RGB.nii.gz')


    print "Current file : hardi snr = " +  str(snr) + ' denoising = ' + str(denoised)
    data, affine, gtab = load_data.get_test_hardi(snr=snr, denoised=denoised)
    FA, RGB = FA_RGB(data, gtab)
    nib.save(nib.Nifti1Image(FA, affine),  'DWIS_hardi-scheme_SNR-' + str(snr) + '_denoising_' + str(denoised) + '_FA.nii.gz')
    nib.save(nib.Nifti1Image(RGB, affine), 'DWIS_hardi-scheme_SNR-' + str(snr) + '_denoising_' + str(denoised) + '_RGB.nii.gz')