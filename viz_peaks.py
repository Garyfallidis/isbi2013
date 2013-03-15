import sys
import nibabel as nib
import numpy as np
from dipy.core.ndindex import ndindex
from dipy.viz import fvtk


def show_peak_directions(peaks, scale=0.3, x=5, y=None, z=None, show_axes=False):
    """ visualize peak directions

    Parameters
    ----------
    peaks : ndarray,
            (X, Y, Z, 15)
    scale : float
            voxel scaling (0 =< `scale` =< 1)
    x : int,
        x slice (0 <= x <= X-1)
    y : int,
        y slice (0 <= y <= Y-1)
    z : int,
        z slice (0 <= z <= Z-1)

    Notes
    -----
    If x, y, z are Nones then the full volume is shown.

    """
    # if x is None and y is None and z is None:
    #    raise ValueError('A slice should be provided')

    pshape = peaks.shape[:-1]
    if x is not None:
        x = np.clip(x, 0, pshape[0] - 1)
    if y is not None:
        y = np.clip(y, 0, pshape[0] - 1)
    if z is not None:
        z = np.clip(z, 0, pshape[0] - 1)

    print x, y, z

    peaks = np.asarray(peaks)
    if peaks.ndim > 4 or peaks.ndim < 4:
        raise ValueError("peaks has wrong shape")

    r = fvtk.ren()

    for index in ndindex(peaks.shape[:-1]):

        peak = peaks[index]
        directions = peak.reshape(peak.shape[0] / 3, 3)

        pos = np.array(index)
        if x is not None:
            pos[0] = x
        if y is not None:
            pos[1] = y
        if z is not None:
            pos[2] = z

        for i in xrange(directions.shape[0]):

            if np.linalg.norm(directions[i]) != 0:

                if (x, y, z) == (None, None, None):
                    line_actor = fvtk.line(index +
                                           scale * np.vstack((-directions[i], directions[i])),
                                           np.abs(directions[i] / np.linalg.norm(directions[i])))
                    fvtk.add(r, line_actor)

                if tuple(pos) == index:
                    line_actor = fvtk.line(pos +
                                           scale * np.vstack((-directions[i], directions[i])),
                                           np.abs(directions[i] / np.linalg.norm(directions[i])))
                    fvtk.add(r, line_actor)

    if show_axes:
        fvtk.add(r, fvtk.axes((2, 2, 2)))
    fvtk.show(r)


if __name__ == '__main__':

    peaks = nib.load(sys.argv[1]).get_data()

    show_peak_directions(peaks, x=None, y=None, z=25, show_axes=True)
