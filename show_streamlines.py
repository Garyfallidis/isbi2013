import numpy as np
from dipy.viz import fvtk


def show_gt_streamlines(streamlines, radii):

    np.random.seed(42)
    colors = np.random.rand(len(streamlines), 3)

    r = fvtk.ren()
    for i in range(len(streamlines)):
        line_actor = fvtk.line(streamlines[i],
                  			   colors[i],
                  			   linewidth=(radii[i, 1] ** 2) / 2.)
        fvtk.add(r, line_actor)
        label_actor = fvtk.label(r, text=str(np.round(radii[i, 1], 2)), 
        		   					pos=(streamlines[i][0]),
        		   					scale=(.8, .8, .8),
        		   					color=(colors[i]) )

        fvtk.add(r, label_actor)

        label_actor_id = fvtk.label(r, text='[' + str(i) + ']', 
        		   					pos=(streamlines[i][-1]),
        		   					scale=(.8, .8, .8),
        		   					color=(colors[i]) )

        fvtk.add(r, label_actor_id)

    fvtk.show(r)


if __name__ == '__main__':


	from load_data import get_train_gt_fibers

	streamlines, radii = get_train_gt_fibers()

	show_gt_streamlines(streamlines, radii)
