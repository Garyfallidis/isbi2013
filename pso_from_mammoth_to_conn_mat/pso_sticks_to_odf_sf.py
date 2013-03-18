import numpy as np
import nibabel as nib

from dipy.data import get_sphere

from scipy.spatial.distance import cdist

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
                for typ in [0, 1, 2, 3]:
                    for ang_t in [0, 10, 20, 30]:
                        for category in ['dti_', 'hardi_']:

                            filename = '{}_pso_sticks_sel={}_NC={}_iso={}_fr={}_Np={}_Ni={}_snr={}_type={}'.format(category, ang_t, NC, iso, fr, Np, Ni, snr, typ)
                            filepath = '/media/Data/work/isbi2013/pso_sticks/' + filename + '.nii.gz'

                            if os.path.exists(filepath):

                                sticks = nib.load(filepath)
                                affine = sticks.get_affine()
                                sticks = sticks.get_data()

                                print(filename)

                                odf_field = np.zeros((50, 50, 50, sphere.vertices.shape[0]))

                                for xx in range(50):
                                    for yy in range(50):
                                        for zz in range(50):
                                            if mask[xx, yy, zz]:

                                                dd = cdist(sphere.vertices, np.vstack((sticks[xx, yy, zz], -sticks[xx, yy, zz])), metric='cosine')

                                                maximas = []

                                                    nan_alert = 0

                                for ii in range(2 * NC):
                                    if ~np.isnan(np.nansum(dd[:, ii])):
                                        maximas.append(np.argsort(dd[:, ii])[0])
                                    else:
                                        nan_alert += 1

                                            odf = np.zeros(sphere.vertices.shape[0])
                                            odf[maximas] = 1 / ((2. * NC) - nan_alert)

                                            odf_field[xx, yy, zz] = odf

                                filename2 = '{}_pso_odf_sf_sel={}_NC={}_iso={}_fr={}_Np={}_Ni={}_snr={}_type={}'.format(category, ang_t, NC, iso, fr, Np, Ni, snr, typ)

                                nib.save(nib.Nifti1Image(odf_field, affine), '/media/Data/work/isbi2013/pso_odf_sf/' + filename2 + '.nii.gz')
