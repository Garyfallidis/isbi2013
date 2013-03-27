import nibabel as nib
import numpy as np

import os.path

# mask=nib.load('/media/Data/work/isbi2013/wm_mask_hardi_01.nii.gz').get_data()
mask=nib.load('/media/Data/work/isbi2013/test_hardi_30_den=1_fa_0025_dilate2_mask.nii.gz').get_data()

iso = 1

for fr in [0,1]:
    for snr in [30,20, 10]:
        for typ in [5,6]:
            for angular_th in [20, 30]:
                for category in ['dti', 'hardi']:

                    Np = 100
                    Ni = 50

                    NC = 1
                    filename = '{}_pso_sticks_sel={}_NC={}_iso={}_fr={}_Np={}_Ni={}_snr={}_type={}'.format(category, 0, NC, iso, fr, Np, Ni, snr, typ)
                    filepath = '/media/Data/work/isbi2013/testing_data_result/pso_sticks/' + filename + '.nii.gz'

                    if os.path.exists(filepath):

                        sticks1 = nib.load(filepath)
                        affine = sticks1.get_affine()
                        sticks1 = sticks1.get_data()

                        NC = 2
                        filename = '{}_pso_sticks_sel={}_NC={}_iso={}_fr={}_Np={}_Ni={}_snr={}_type={}'.format(category, 0, NC, iso, fr, Np, Ni, snr, typ)
                        sticks2 = nib.load('/media/Data/work/isbi2013/testing_data_result/pso_sticks/' + filename + '.nii.gz').get_data()

                        NC = 3
                        filename = '{}_pso_sticks_sel={}_NC={}_iso={}_fr={}_Np={}_Ni={}_snr={}_type={}'.format(category, 0, NC, iso, fr, Np, Ni, snr, typ)
                        sticks3 = nib.load('/media/Data/work/isbi2013/testing_data_result/pso_sticks/' + filename + '.nii.gz').get_data()

                        peak_field = np.zeros((50, 50, 50, 3, 3))

                        changed_to_2 = 0
                        changed_to_1 = 0

                        print(filename)

                        for xx in range(50):
                            if (xx % 10) == 0:
                                print(xx)
                            for yy in range(50):
                                for zz in range(50):

                                    if mask[xx,yy,zz]:

                                        peaks = sticks3[xx, yy, zz]

                                        ang01 = 90 - np.abs(np.arccos(np.inner(peaks[0], peaks[1])) * (180 / np.pi) - 90)
                                        ang02 = 90 - np.abs(np.arccos(np.inner(peaks[0], peaks[2])) * (180 / np.pi) - 90)
                                        ang12 = 90 - np.abs(np.arccos(np.inner(peaks[1], peaks[2])) * (180 / np.pi) - 90)

                                        if ((ang01 < angular_th) or (ang02 < angular_th) or (ang12 < angular_th)):
                                            #try 2
                                            peaks = sticks2[xx, yy, zz]

                                            ang01 = 90 - np.abs(np.arccos(np.inner(peaks[0], peaks[1])) * (180 / np.pi) - 90)

                                            if (ang01 < angular_th):
                                                #keep 1
                                                peak_field[xx, yy, zz, 0] = sticks1[xx, yy, zz]
                                                changed_to_1 += 1
                                            else:
                                                #keep 2
                                                peak_field[xx, yy, zz, :2] = sticks2[xx, yy, zz]
                                                changed_to_2 += 1
                                        else:
                                            #keep 3
                                            peak_field[xx, yy, zz] = sticks3[xx, yy, zz]

                        print('changed to 2: {},   changed to 1: {}'.format(changed_to_2, changed_to_1))
                        filename2 = '{}_pso_sticks_sel={}_NC={}_iso={}_fr={}_Np={}_Ni={}_snr={}_type={}'.format(category, angular_th, NC, iso, fr, Np, Ni, snr, typ)

                        nib.save(nib.Nifti1Image(peak_field, affine), '/media/Data/work/isbi2013/testing_data_result/pso_sticks/' + filename2 + '.nii.gz')
