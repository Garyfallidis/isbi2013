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
                
        for i in xrange(directions.shape[0]):
            
            if np.linalg.norm(directions[i]) != 0:
                
                x = directions[i][0]
                y = directions[i][1]
                z = directions[i][2]
                #print x,y,z

                #i = find_where( sphere.vertices == x,y,z )
                #sf[i] = 1

        SF[index] = sf

    return SF

                
                

if __name__ == '__main__':
    peaks = nib.load('peaks.nii.gz').get_data()
    refaff = nib.load('peaks.nii.gz').get_affine()

    vertices = np.loadtxt('sphere724.txt')
    sphere = Sphere(xyz=vertices)

    #print vertices
    ijk = vertices[700]
    #i = nonzeros( vertices == np.array([9.98801772e-01,4.14364641e-03,4.87632000e-02]) )
    print ijk

    print vertices[vertices == ijk]
    print -ijk
    print vertices[vertices == -ijk]
    print np.nonzero( vertices == ijk )[0][0]
    
    SF = peaks_to_sf(peaks, sphere)
    
    SF_img = nib.Nifti1Image(SF.astype(np.float32), refaff)
    nib.save(SF_img, 'sf_peaks.nii')


    

