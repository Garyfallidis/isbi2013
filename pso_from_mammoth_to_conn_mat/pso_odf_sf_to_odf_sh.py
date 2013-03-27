import numpy as np
import nibabel as nib

from dipy.data import get_sphere

from dipy.reconst.shm import sf_to_sh

import os.path

#fixed method parameters
Np = 100
Ni = 50

mask = nib.load('wm_mask_hardi_01.nii.gz').get_data()

sphere = get_sphere('symmetric724')

for NC in [1, 2, 3]:
    for iso in [0, 1]:
        for fr in [0, 1]:
            for snr in [30, 10]:
                for typ in [1]:
                    for ang_t in [23,33,25,35]:#[0, 10, 20, 30]:
                        for category in ['dti', 'hardi']:

							filename = '{}_pso_odf_sf_sel={}_NC={}_iso={}_fr={}_Np={}_Ni={}_snr={}_type={}'.format(category, ang_t, NC, iso, fr, Np, Ni, snr, typ)
							filepath = '/media/Data/work/isbi2013/pso_odf_sf/' + filename + '.nii.gz'

							if os.path.exists(filepath):
								odf = nib.load(filepath)
								affine = odf.get_affine()
								odf = odf.get_data()

								print(filename)

								odf_sh = sf_to_sh(odf, sphere, sh_order=8,basis_type='mrtrix')

								filename2 = '{}_pso_odf_sh_sel={}_NC={}_iso={}_fr={}_Np={}_Ni={}_snr={}_type={}'.format(category, ang_t, NC, iso, fr, Np, Ni, snr, typ)

								nib.save(nib.Nifti1Image(odf_sh, affine), '/media/Data/work/isbi2013/pso_odf_sh/' + filename2 + '.nii.gz')




