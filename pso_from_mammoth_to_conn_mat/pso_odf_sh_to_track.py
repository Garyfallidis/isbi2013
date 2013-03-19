import numpy as np
import nibabel as nib

from dipy.data import get_sphere

from subprocess import Popen, PIPE

import os.path


def pipe(cmd):
    """ A tine pipeline system to run external tools.

    For more advanced pipelining use nipype http://www.nipy.org/nipype
    """
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    sto = p.stdout.readlines()
    ste = p.stderr.readlines()
    print(sto)
    print(ste)

#fixed method parameters
Np = 100
Ni = 50

sphere = get_sphere('symmetric724')

for NC in [1, 2, 3]:
    for iso in [0, 1]:
        for fr in [0, 1]:
            for snr in [30, 10]:
                for typ in [0, 1, 2, 3]:
                    for ang_t in [0, 10, 20, 30]:
                        for category in ['dti', 'hardi']:

                            filename = '{}_pso_odf_sh_sel={}_NC={}_iso={}_fr={}_Np={}_Ni={}_snr={}_type={}'.format(category, ang_t, NC, iso, fr, Np, Ni, snr, typ)
                            filepath = '/media/Data/work/isbi2013/pso_odf_sh/' + filename + '.nii.gz'

                            if os.path.exists(filepath):

                            	print(filename)

                            	trackfile = '{}_pso_track_sel={}_NC={}_iso={}_fr={}_Np={}_Ni={}_snr={}_type={}'.format(category, ang_t, NC, iso, fr, Np, Ni, snr, typ)

                            	direc = '/media/Data/work/isbi2013/'

                            	seeds_per_vox = 5
                            	num_of_cpus = 3

                            	cmd = 'python /media/Data/work/scilpy/scripts/stream_local.py -odf '  + filepath + ' -m ' + direc + 'wm_mask_hardi_01.nii.gz -s data/training-data_rois.nii.gz -n -' + str(seeds_per_vox) +' -process ' + str(num_of_cpus) + ' -o ' + direc + 'pso_track/' + trackfile + '.trk'

                            	pipe(cmd)