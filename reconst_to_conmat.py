
import nibabel as nib

from subprocess import Popen, PIPE

from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.dsi import DiffusionSpectrumDeconvModel
from dipy.data import get_sphere
from dipy.viz.mayavi.spheres import show_odfs
from dipy.reconst.shm import sf_to_sh

from load_data import get_train_dsi, get_train_rois, get_train_mask
from show_streamlines import show_streamlines
from conn_mat import connectivity_matrix

from time import time



def pipe(cmd):
    """ A tine pipeline system to run external tools.

    For more advanced pipelining use nipype http://www.nipy.org/nipype
    """
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    sto = p.stdout.readlines()
    ste = p.stderr.readlines()
    print(sto)
    print(ste)

data, affine, gtab = get_train_dsi(30)
mask, affine = get_train_mask()

# data = data[25 - 10:25 + 10, 25 - 10:25 + 10, 25]
# data = data[:, :, 25]

# model = GeneralizedQSamplingModel(gtab,
#                                   method='gqi2',
#                                   sampling_length=3,
#                                   normalize_peaks=False)

model = DiffusionSpectrumDeconvModel(gtab)

t0 = time()

fit = model.fit(data, mask)

sphere = get_sphere('symmetric724')

odf = fit.odf(sphere)

t1 = time()

print t1 - t0

odf_sh = sf_to_sh(odf, sphere, sh_order=8,
                  basis_type='mrtrix')

nib.save(nib.Nifti1Image(odf_sh, affine), 'odf_sh.nii.gz')

cmd = 'python ~/Devel/scilpy/scripts/stream_local.py -odf odf_sh.nii.gz -m data/training-data_mask.nii.gz -s data/training-data_rois.nii.gz -n -1 -process 1 -o streams.trk -maximum'
pipe(cmd)

streams, hdr = nib.trackvis.read('streams.trk')

streamlines = [s[0] for s in streams]

show_streamlines(streamlines, opacity=0.5)

rois, affine = get_train_rois()

mat, srois = connectivity_matrix(streamlines, rois)

golden_mat = np.load('train_connmat.npy')

mat += mat.T

golden_mat += golden_mat.T
