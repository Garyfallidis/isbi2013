import nibabel as nib
import numpy as np

import os.path

Np = 100
Ni = 50

for NC in [1, 2, 3]:
    for iso in [0, 1]:
        for fr in [0, 1]:
            for snr in [30, 10]:
                for typ in [0, 1, 2, 3]:
                    for category in ['dti_', 'hardi_']:
                        volume = np.zeros((50, 50, 50, NC, 3))
                        for slicez in range(50):

                            filename = '{}_pso_sticks_slice={}_NC={}_iso={}_fr={}_Np={}_Ni={}_snr={}_type={}'.format(category, slicez, NC, iso, fr, Np, Ni, snr, typ)
                            filepath = '/media/Data/work/isbi2013/pso_sticks_slices/' + filename + '.nii.gz'

                            cont = os.path.exists(filepath)
                            if cont:
                                data = nib.load(filepath)
                                affine = data.get_affine()
                                data = data.get_data()

                                volume[:, :, slicez, :, :] = data

                        if cont:
                            filename = '{}_pso_sticks_sel={}_NC={}_iso={}_fr={}_Np={}_Ni={}_snr={}_type={}'.format(category, 0, NC, iso, fr, Np, Ni, snr, typ)
                            nib.save(nib.Nifti1Image(volume, affine), '/media/Data/work/isbi2013/pso_sticks/' + filename + '.nii.gz')
                            print('saved ' + filename)
