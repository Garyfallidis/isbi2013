from load_data import get_train_dti, get_train_hardi, get_train_dsi, get_train_mask
from dipy.reconst.dti import TensorModel
from pylab import imshow, show, colorbar, subplot, title, figure
import nibabel as nib
import numpy as np


for datat in range(2):

	if datat == 0:
		print 'fitting with dti'
		data, affine, gtab = get_train_dti(30)
	elif datat == 1:
		print 'fitting with hardi'
		data, affine, gtab = get_train_hardi(30)
	elif datat == 2:
		print 'fitting with dsi'
		data, affine, gtab = get_train_dsi(30)

	mask, affine = get_train_mask()

	data.shape
	mask.shape


	model = TensorModel(gtab)
	fit = model.fit(data, mask)
	print 'done!'
	fa = fit.fa


	slice_z = 25

	Th = [0.05, 0.075, 0.1,0.15]

	figure(2*datat+1)
	imshow(fa[:, :, slice_z], interpolation='nearest')
	colorbar()
	title(mask.sum())

	figure(2*datat + 2)
	for i in range(4):
	    subplot(2, 2, i + 1)
	    tmp = fa > Th[i]
	    imshow(tmp[:, :, slice_z], interpolation='nearest')
	    title((Th[i], tmp.sum()))


show()




# nib.save(nib.Nifti1Image((fa > 0.05), affine), 'wm_mask.nii.gz')
