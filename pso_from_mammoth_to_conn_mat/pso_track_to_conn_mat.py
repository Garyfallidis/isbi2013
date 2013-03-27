import numpy as np
import nibabel as nib

from show_streamlines import show_streamlines

from load_data import get_train_rois

from conn_mat import connectivity_matrix

from dipy.io.pickles import save_pickle

import os.path


def streams_to_connmat(filename, seeds_per_voxel=5, thr=[0.25, 0.5, 0.75]):

    streams, hdr = nib.trackvis.read(filename)
    streamlines = [s[0] for s in streams]

    #show_streamlines(streamlines, opacity=0.5)

    rois, affine = get_train_rois()

    mat, srois, ratio = connectivity_matrix(streamlines, rois)

    golden_mat = np.load('train_connmat.npy')

    lr = []
    for i in range(1, 41):
        lr.append(np.sum(rois==i))

    lr = seeds_per_voxel * np.array(lr, dtype='f8')

    mat /= lr

    mat += mat.T

    golden_mat += golden_mat.T

    conn_mats = []
    diffs = []
    for th in thr:
        conn_mat = mat > th
        conn_mats.append(conn_mat)
        diffs.append(np.sum(np.abs(conn_mat-golden_mat)))

    return mat, conn_mats, diffs


Np = 100
Ni = 50



for NC in [1, 2, 3]:
    for iso in [0, 1]:
        for fr in [0, 1]:
            for snr in [30, 10]:
                for typ in [0, 1, 2, 3]:
                    for ang_t in [20,23,25, 30,33,35]:
                        for category in ['dti']:#, 'hardi']:

                            filename = '{}_pso_track_sel={}_NC={}_iso={}_fr={}_Np={}_Ni={}_snr={}_type={}'.format(category, ang_t, NC, iso, fr, Np, Ni, snr, typ)
                            filepath = '/media/Data/work/isbi2013/pso_track/' + filename + '.trk'

                            if os.path.exists(filepath):

                                mat, conn_mats, diffs = streams_to_connmat(filepath, seeds_per_voxel=5, thr=[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])#thr=[0.25, 0.5, 0.75])

                                print(filename, diffs)

                                filename2 = '{}_pso_conn_mat_sel={}_NC={}_iso={}_fr={}_Np={}_Ni={}_snr={}_type={}'.format(category, ang_t, NC, iso, fr, Np, Ni, snr, typ)
                                save_pickle('/media/Data/work/isbi2013/pso_conn_mat/' + filename2 + '.pkl', {'mat':mat, 'conn_mats':conn_mats, 'diffs':diffs})

