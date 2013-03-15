import nibabel as nib
import numpy as np

Np=100
Ni=50

for NC in [2,3]:#####
	for iso in [0,1]:
		for fr in [0,1]:#####
			for snr in [30,10]:
				for den in [0,1]:
					volume = np.zeros((50,50,50,NC,3))
					for slicez in range(50):

						filename = 'sticks_NC={}_iso={}_fr={}_Np={}_Ni={}_snr={}_den={}_slic={}'.format(NC,iso,fr,Np,Ni,snr,den,slicez)

						data = nib.load('/media/Data/work/isbi2013/slices_out_pso/' + filename + '.nii.gz')
						affine = data.get_affine()
						data=data.get_data()

						volume[:,:,slicez,:,:] = data

					filename = '3D_sticks_NC={}_iso={}_fr={}_Np={}_Ni={}_snr={}_den={}'.format(NC,iso,fr,Np,Ni,snr,den)
					nib.save(nib.Nifti1Image(volume, affine), '/media/Data/work/isbi2013/slices_out_pso/' + filename + '.nii.gz')
					print('saved ' + filename)