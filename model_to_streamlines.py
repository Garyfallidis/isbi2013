import nibabel as nib
from dipy.reconst.dti import TensorModel
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.dsi import DiffusionSpectrumDeconvModel
from dipy.data import get_sphere
from load_data import (get_train_dsi,
                       get_train_mask,
                       get_train_rois)
from dipy.tracking.eudx import EuDX
from dipy.reconst.odf import peaks_from_model


data, affine, gtab = get_train_dsi(30)
mask, _ = get_train_mask()
rois, _ = get_train_rois()

# data = data[25 - 10:25 + 10, 25 - 10:25 + 10, 25]
# data = data[:, :, 25]

fa_mask = TensorModel(gtab).fit(data, mask).fa > 0.1


gqi_model = GeneralizedQSamplingModel(gtab,
                                      method='gqi2',
                                      sampling_length=3,
                                      normalize_peaks=False)

gqi_fit = gqi_model.fit(data, mask)

sphere = get_sphere('symmetric724')

peaks = peaks_from_model(gqi_model, data, sphere, 0.35, 30,
                         mask=fa_mask, normalize_peaks=True)


seeds = np.vstack(np.where((rois == 1) | (rois == 2))).T
seeds = np.ascontiguousarray(seeds)

eu = EuDX(peaks.peak_values, peaks.peak_indices,
          seeds=seeds, odf_vertices=sphere.vertices)

from dipy.tracking.metrics import length

streamlines = [s for s in eu if length(s) > 10]

from dipy.viz import fvtk

r = fvtk.ren()

from dipy.viz.colormap import line_colors

#from dipy.segment.quickbundles import QuickBundles

#qb = QuickBundles(streamlines, 10., 18)

#fvtk.add(r, fvtk.line(qb.centroids,
#                      line_colors(qb.centroids)))

fvtk.add(r, fvtk.line(streamlines,
                      line_colors(streamlines)))

from show_streamlines import show_gt_streamlines

from load_data import get_train_gt_fibers

streamlines_gt, radii_gt = get_train_gt_fibers()

streamlines_gt = [s + np.array([24.5, 24.5, 24.5]) for s in streamlines_gt]

show_gt_streamlines(streamlines_gt, radii_gt, r=r)

fvtk.show(r)
