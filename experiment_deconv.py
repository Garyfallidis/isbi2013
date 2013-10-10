import numpy as np
import nibabel as nib

from dipy.reconst.dti import fractional_anisotropy
from dipy.reconst.dti import TensorModel
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.dsi import DiffusionSpectrumDeconvModel
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, ConstrainedSDTModel
from dipy.data import get_sphere
from dipy.reconst.shm import sf_to_sh

from load_data import get_jc_hardi, get_test_mask, get_test_hardi
from show_streamlines import show_streamlines
from conn_mat import connectivity_matrix

from dipy.io.pickles import save_pickle, load_pickle

from time import time


threshold = 0.78
Noise = 0
if __name__ == '__main__':
    data, affine, gtab = get_test_hardi(snr=Noise, denoised=0)
    mask = get_test_mask()
    
    #print(data.min())
    if data.max() <= 1 :
        if data.min() != 0 :
            data = data / data.min()
        else:
            data = data * 1000
    
    #print(data.min())
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask)
    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    indices = np.where(FA > threshold)
    print(FA[mask].max())
    
    nib.save(nib.Nifti1Image(FA.astype('float32'), affine), 
             'FA.nii.gz')

    lambdas = tenfit.evals[indices][:, :2]
    S0s = data[indices][:, 0]
    S0 = np.mean(S0s)
    if S0 == 0 :
        S0 = 1        
    l01 = np.mean(lambdas, axis = 0) 
    evals = np.array([l01[0], l01[1], l01[1]])
    print evals, l01[1] / l01[0], S0
    ratio = l01[1] / l01[0]

    from dipy.data import get_sphere
    sphere = get_sphere('symmetric724')

    csd_model = ConstrainedSphericalDeconvModel(gtab, (evals, S0))
    from dipy.reconst.odf import peaks_from_model
    peaks = peaks_from_model(model=csd_model,
                             data=data,
                             sphere=sphere,
                             relative_peak_threshold=0.25,
                             min_separation_angle=45,
                             mask=mask,
                             return_odf=False, 
                             return_sh=True, 
                             normalize_peaks=False,
                             sh_order=8,
                             sh_basis_type='mrtrix',
                             npeaks=5, 
                             parallel=True, nbr_process=8)

    

    shm_coeff = peaks.shm_coeff
    nib.save(nib.Nifti1Image(shm_coeff.astype('float32'), affine), 
             'fodf_csd_sh' + str(threshold) + '_snr' + str(Noise) + '.nii.gz')

    myPeaksDirs = peaks.peak_dirs
    test = np.reshape(myPeaksDirs, [myPeaksDirs.shape[0], 
                                    myPeaksDirs.shape[1], 
                                    myPeaksDirs.shape[2], 
                                    myPeaksDirs.shape[3]*myPeaksDirs.shape[4]])    
    nib.save(nib.Nifti1Image(test.astype('float32'), affine), 
             'peaks_csd' + str(threshold) + '_snr' + str(Noise) +  '.nii.gz') 

    sdt_model = ConstrainedSDTModel(gtab, ratio)
    from dipy.reconst.odf import peaks_from_model
    peaks = peaks_from_model(model=sdt_model,
                             data=data,
                             sphere=sphere,
                             relative_peak_threshold=0.25,
                             min_separation_angle=45,
                             mask=mask,
                             return_odf=False, 
                             return_sh=True, 
                             normalize_peaks=False,
                             sh_order=8,
                             sh_basis_type='mrtrix',
                             npeaks=5, 
                             parallel=True, nbr_process=8)
    

    shm_coeff = peaks.shm_coeff
    nib.save(nib.Nifti1Image(shm_coeff.astype('float32'), affine), 
             'fodf_sdt_sh' + str(threshold) + '_snr' + str(Noise) +  '.nii.gz')

    myPeaksDirs = peaks.peak_dirs
    test = np.reshape(myPeaksDirs, [myPeaksDirs.shape[0], 
                                    myPeaksDirs.shape[1], 
                                    myPeaksDirs.shape[2], 
                                    myPeaksDirs.shape[3]*myPeaksDirs.shape[4]])    
    nib.save(nib.Nifti1Image(test.astype('float32'), affine), 
             'peaks_sdt' + str(threshold) + '_snr' + str(Noise) +  '.nii.gz') 
