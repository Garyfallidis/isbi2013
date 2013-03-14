import numpy as np
from dipy.tracking.vox2track import track_counts


def connectivity_matrix(streamlines, rois):
    tcs, tes = track_counts(streamlines, rois.shape, return_elements=True)
    srois = {}

    for i in range(1, 41):
        roi_indices = np.array(np.where(rois == i)).T
        for roind in roi_indices:
            try:
                for s in tes[tuple(roind)]:
                    try:
                        srois[s].append(i - 1)
                    except KeyError:
                        srois[s] = [i - 1]
            except KeyError:
                pass

    mat = np.zeros((40, 40))
    for s in srois:
        srois[s] = set(srois[s])
        if len(srois[s]) >= 3:            
            continue
        if len(srois[s]) < 2:            
            continue

        try:
            mat[tuple(srois[s])] += 1
        except IndexError:
            print IndexError
            print srois[s]

    return mat, srois, 100 * mat.sum() / np.float(len(streamlines))

 