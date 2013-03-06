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
        mat[tuple(srois[s])] += 1

    return mat, srois


    # def connectivity_matrix2(streamlines, rois):        
    #     srois = {}
    #     for i in range(1, 41):
    #         roi_indices = np.array(np.where(rois == i)).T
    #         for si, s in enumerate(streamlines):
    #             first = (np.floor(s[0]) + .5).astype(np.int)
    #             last = (np.floor(s[-1]) + .5).astype(np.int)

    #             for roind in roi_indices:
    #                 try:
    #                     if tuple(roind) == tuple(first):
    #                         srois[si].append(i - 1)                        
    #                 except KeyError:
    #                     srois[si] = [i - 1]

    #     return srois
    #     mat = np.zeros((40, 40))
    #     for s in srois:
    #         srois[s] = set(srois[s])
    #         mat[tuple(srois[s])] += 1

    #     return mat, srois
 