import numpy as np
import nibabel as nib

from dipy.reconst.dti import fractional_anisotropy
from dipy.reconst.dti import TensorModel
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.dsi import DiffusionSpectrumDeconvModel
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, ConstrainedSDTModel
from dipy.data import get_sphere
from dipy.reconst.shm import sf_to_sh

from load_data import get_jc_hardi, get_test_mask, get_test_wm_mask
from show_streamlines import show_streamlines
from conn_mat import connectivity_matrix

from dipy.io.pickles import save_pickle, load_pickle

from time import time

threshold = 0.5
from dipy.data import get_sphere
sphere = get_sphere('symmetric724')
dname = 'SNR20/'

if __name__ == '__main__':
    data, affine, gtab = get_jc_hardi(20)    
    mask = get_test_mask()
    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data, mask)
    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    nib.save(nib.Nifti1Image(FA.astype('float32'), affine), 
             'FA.nii.gz')
    
    for i in range(27) :
        print 'White matter bundle: ', i
        wm_mask = get_test_wm_mask(i)
        print(FA[wm_mask].max())
        indicesAniso = np.where(np.logical_and(FA > threshold, wm_mask))  

        print '    Response function'
        S0s = data[indicesAniso][:, np.nonzero(gtab.b0s_mask)[0]]                               
        S0 = np.mean(S0s)
        if S0 == 0 :
            S0 = 1       
        lambdas = tenfit.evals[indicesAniso][:, :2]   

        mean_evals = np.mean(lambdas, axis=0)
        evals = np.array([mean_evals[0], mean_evals[1], mean_evals[1]])
        response = (evals, S0)
        ratio = mean_evals[1] / mean_evals[0]

        print '    ... Valued ', response
        
        response_ar = np.array([mean_evals[0], mean_evals[1], mean_evals[1], S0])
        if i < 10 :
            np.savetxt(dname + 'response_0' + str(i) + '.txt', response_ar)
        else :
            np.savetxt(dname +  'response_' + str(i) + '.txt', response_ar)

        csd_model = ConstrainedSphericalDeconvModel(gtab, (evals, S0))
        from dipy.reconst.odf import peaks_from_model
        peaks = peaks_from_model(model=csd_model,
                                 data=data,
                                 sphere=sphere,
                                 relative_peak_threshold=0.25,
                                 min_separation_angle=45,
                                 mask=wm_mask,
                                 return_odf=False, 
                                 return_sh=True, 
                                 normalize_peaks=False,
                                 sh_order=8,
                                 sh_basis_type='mrtrix',
                                 npeaks=5, 
                                 parallel=True, nbr_process=8)

    

        shm_coeff = peaks.shm_coeff
        
        
        myPeaksDirs = peaks.peak_dirs
        test = np.reshape(myPeaksDirs, [myPeaksDirs.shape[0], 
                                        myPeaksDirs.shape[1], 
                                        myPeaksDirs.shape[2], 
                                        myPeaksDirs.shape[3]*myPeaksDirs.shape[4]])    
        if i < 10 : 
            nib.save(nib.Nifti1Image(shm_coeff.astype('float32'), affine), 
                     dname + 'fodf_csd_sh_0' + str(i) + '.nii.gz')
            nib.save(nib.Nifti1Image(test.astype('float32'), affine), 
                     dname +  'peaks_csd._0' + str(i) + '.nii.gz') 
        else :
            nib.save(nib.Nifti1Image(shm_coeff.astype('float32'), affine), 
                     dname +  'fodf_csd_sh_' + str(i) + '.nii.gz')
            nib.save(nib.Nifti1Image(test.astype('float32'), affine), 
                     dname +  'peaks_csd._' + str(i) + '.nii.gz') 

        sdt_model = ConstrainedSDTModel(gtab, ratio)
        from dipy.reconst.odf import peaks_from_model
        peaks = peaks_from_model(model=sdt_model,
                                 data=data,
                                 sphere=sphere,
                                 relative_peak_threshold=0.25,
                                 min_separation_angle=45,
                                 mask=wm_mask,
                                 return_odf=False, 
                                 return_sh=True, 
                                 normalize_peaks=False,
                                 sh_order=8,
                                 sh_basis_type='mrtrix',
                                 npeaks=5, 
                                 parallel=True, nbr_process=8)
        
        
        shm_coeff = peaks.shm_coeff
        myPeaksDirs = peaks.peak_dirs
        test = np.reshape(myPeaksDirs, [myPeaksDirs.shape[0], 
                                        myPeaksDirs.shape[1], 
                                        myPeaksDirs.shape[2], 
                                        myPeaksDirs.shape[3]*myPeaksDirs.shape[4]])    
        if i < 10 : 
            nib.save(nib.Nifti1Image(shm_coeff.astype('float32'), affine), 
                     dname + 'fodf_sdt_sh_0' + str(i) + '.nii.gz')
            nib.save(nib.Nifti1Image(test.astype('float32'), affine), 
                     dname + 'peaks_sdt._0' + str(i) + '.nii.gz') 
        else :
            nib.save(nib.Nifti1Image(shm_coeff.astype('float32'), affine), 
                     dname + 'fodf_sdt_sh_' + str(i) + '.nii.gz')
            nib.save(nib.Nifti1Image(test.astype('float32'), affine), 
                     dname +  'peaks_sdt._' + str(i) + '.nii.gz') 



