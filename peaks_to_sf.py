import nibabel as nib
import numpy as np
from dipy.core.ndindex import ndindex
from dipy.core.sphere import Sphere
from dipy.viz import fvtk


def peaks_to_sf(peaks, sphere):
    """ Peaks to spherical function (SF)

    Parameters
    ----------
    peaks : ndarray,
            (X, Y, Z, 15)
    sphere : Sphere
            The points on which to peaks were sampled.

    """

    pshape = peaks.shape[:-1]
    peaks = np.asarray(peaks)
    if peaks.ndim > 4 or peaks.ndim < 4:
        raise ValueError("peaks has wrong shape")

    SF = np.zeros( (pshape + (sphere.vertices.shape[0],)))
    for index in ndindex(peaks.shape[:-1]):
        peak = peaks[index]
        directions = peak.reshape(peak.shape[0] / 3, 3)
        sf = np.zeros((sphere.vertices.shape[0]))
        
        angles = np.abs(np.dot(directions, sphere.vertices.T))                                
        
        i = 0
        for a in angles:
            if np.linalg.norm(a) != 0:
                j = np.argsort(a)[-2:]
                sf[j]=1*np.linalg.norm(directions[i])

            i += 1
#        if index[0] == 0 and index[1] == 0 and index[2] == 0 :            
#             from dipy.viz import fvtk
#             r=fvtk.ren()
#             fvtk.add(r, fvtk.sphere_funcs(sf+0.5, sphere))
#             #fvtk.add(r, fvtk.axes())
#             fvtk.show(r)

        SF[index] = sf

    return SF
                

if __name__ == '__main__':
    peaks = nib.load('peaks.nii.gz').get_data()
    refaff = nib.load('peaks.nii.gz').get_affine()

    vertices = np.loadtxt('sphere724.txt')
    sphere = Sphere(xyz=vertices)    
    SF = peaks_to_sf(peaks, sphere)
    
    SF_img = nib.Nifti1Image(SF.astype(np.float), refaff)
    nib.save(SF_img, 'sf_peaks.nii')



    

