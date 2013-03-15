import nibabel as nib
import numpy as np
from dipy.sims.voxel import MultiTensor
from load_data import get_train_dti


mask = nib.load('wm_mask_hardi_01.nii.gz').get_data()

_, affine, gtab = get_train_dti()


def parse_params(params, NC, iso, fr):
    mevals = []
    angles = []
    fractions = []
    for i in range(NC):
        mevals.append(np.array([params[4 * i], params[4 * i + 1], params[4 * i + 1]]))
        angles.append(np.array([params[4 * i + 2], params[4 * i + 3]]))
    if iso:
        mevals.append(np.array([params[4 * NC], params[4 * NC], params[4 * NC]]))
        angles.append(np.array([0., 0.]))
        iso_frac = params[4 * NC + 1]
    if ~fr:
        if ~iso:
            fractions.append(np.ones((1, NC)) * (100. / NC))
        else:
            fractions.append(np.vstack((np.ones((NC, 1)) * (1. / NC) * (100. - iso_frac), np.array([iso_frac]))).T)
    else:
        if NC == 1:
            frac = 100.
        else:
            fracc = params[-(NC - 1)]
            frac = np.zeros(NC)
            frac[0] = fracc[0]
            if NC == 2:
                frac[1] = 100. - frac[0]
            else:
                frac[1] = (100. - frac[0]) * fracc[1] / 100.
            if NC == 3:
                frac[2] = 100. - (frac[0] + frac[1])
        if ~iso:
            fractions.append(frac)
        else:
            fractions.append(np.vstack((frac * (100. - iso_frac) / 100., np.array([iso_frac]))).T)

    return np.array(mevals), np.array(angles), np.array(fractions).T


f = open('out__pso_testDataSlice__23', 'r')
# f = open('out__pso_testDataSlice__23', 'r')

line = f.readline()
while(line!=''):
	# NC iso fr Npart Niter den
	params = np.array(line[:-1].split(' '), dtype=np.int)
	sliceXY = np.zeros((50,50,params[0],3))

	#first line of data
	line = f.readline()
	dat = np.array(line[:-1].split(' '),dtype=np.float64)

	#number of voxel in slice
	slicez = int(dat[2])
	nb_to_read = int(mask[:,:,slicez].sum())

	#parse first voxel
	mevals, angles, fractions = parse_params(dat[3:],params[0], params[1], params[2])

	# could keep fractions also
	_, sticks = MultiTensor(gtab, mevals, 100, angles, fractions, None)
	if params[1]:
		sticks = sticks[:-1]

	#save sticks
	sliceXY[dat[0],dat[1]] = sticks

	# read rest of slice
	for i in range(nb_to_read-1):

		line = f.readline()
		dat = np.array(line[:-1].split(' '),dtype=np.float64)

		mevals, angles, fractions = parse_params(dat[3:],params[0], params[1], params[2])

		_, sticks = MultiTensor(gtab, mevals, 100, angles, fractions, None)
		if params[1]:
			sticks = sticks[:-1]

		sliceXY[dat[0],dat[1]] = sticks

	filename = 'sticks_NC={}_iso={}_fr={}_Np={}_Ni={}_snr={}_den={}_slic={}'.format(params[0],params[1],params[2],params[3],params[4],params[5],params[6],slicez)
	nib.save(nib.Nifti1Image(sliceXY, affine), '/media/Data/work/isbi2013/slices_out_pso/' + filename + '.nii.gz')

	print('saved! NC = {} iso = {} fr = {} snr = {} den = {} slic = {}'.format(params[0],params[1],params[2],params[5],params[6],slicez))

	line = f.readline()

f.close()


