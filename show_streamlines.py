import numpy as np
from dipy.viz import fvtk
from dipy.viz.colormap import line_colors


def show_gt_streamlines(streamlines, radii, cmap='orient', r=None):

    if cmap is None:
        np.random.seed(42)
        colors = np.random.rand(len(streamlines), 3)
    if cmap == 'orient':
        colors = line_colors(streamlines)

    if r is None:
        ren = fvtk.ren()
    else:
        ren = r

    for i in range(len(streamlines)):
        line_actor = fvtk.line(streamlines[i],
                               colors[i],
                               linewidth=(radii[i, 1] ** 2) / 2.)
        fvtk.add(ren, line_actor)
        label_actor = fvtk.label(ren, text=str(np.round(radii[i, 1], 2)),
                                 pos=(streamlines[i][0]),
                                 scale=(.8, .8, .8),
                                 color=(colors[i]))

        fvtk.add(ren, label_actor)

        label_actor_id = fvtk.label(ren, text='[' + str(i) + ']',
                                    pos=(streamlines[i][-1]),
                                    scale=(.8, .8, .8),
                                    color=(colors[i]))

        fvtk.add(ren, label_actor_id)

    if r is None:
        fvtk.show(ren)
    else:
        return ren


if __name__ == '__main__':

    from load_data import get_train_gt_fibers

    streamlines, radii = get_train_gt_fibers()

    show_gt_streamlines(streamlines, radii, 'orient', None)

    from load_data import get_train_rois

    rois, affine = get_train_rois()

    streamlines = [s + np.array([24.5, 24.5, 24.5]) for s in streamlines]

    from conn_mat import connectivity_matrix
    
    mat, srois = connectivity_matrix(streamlines, rois)
    
    

