
import nibabel as nib
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.dsi import DiffusionSpectrumDeconvModel
from dipy.data import get_sphere
from dipy.viz.mayavi.spheres import show_odfs
from load_data import get_train_dsi, get_train_rois, get_train_mask


data, affine, gtab = get_train_dsi(30)
mask, affine = get_train_mask()


# data = data[25 - 10:25 + 10, 25 - 10:25 + 10, 25]
# data = data[:, :, 25]

gqi_model = GeneralizedQSamplingModel(gtab,
                                      method='gqi2',
                                      sampling_length=3,
                                      normalize_peaks=False)

gqi_fit = gqi_model.fit(data, mask)

sphere = get_sphere('symmetric724')

gqi_odf = gqi_fit.odf(sphere)

from dipy.reconst.shm import sf_to_sh

gqi_sh = sf_to_sh(gqi_odf, sphere, sh_order=8,
                  basis_type='mrtrix')

nib.save(nib.Nifti1Image(gqi_sh, affine), 'gqi_sh.nii.gz')

"""
python ~/Devel/scilpy/scripts/stream_local.py -odf gqi_sh.nii.gz -m data/training-data_mask.nii.gz -s data/training-data_rois.nii.gz -n -1 -process 1 -o gqi_streams.trk

python ~/Devel/scilpy/scripts/stream_local.py -odf gqi_sh.nii.gz -m data/training-data_mask.nii.gz -s data/training-data_rois.nii.gz -n -1 -process 1 -o gqi_streams.trk -maximum

"""

from dipy.viz import fvtk

r= fvtk.ren()

streams, hdr = nib.trackvis.read('gqi_streams.trk')

streamlines = [s[0] for s in streams]

fvtk.add(r, fvtk.line(streamlines, fvtk.red))

fvtk.show(r)