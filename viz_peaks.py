import numpy as np
from dipy.core.ndindex import ndindex
from dipy.viz import fvtk

peaks = np.zeros((50, 50, 1, 15))

dummy = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).ravel()

peaks[..., :9] = dummy[:]

def crossings(peaks, scale = 0.3):

    r = fvtk.ren()
    for index in ndindex(peaks.shape[:-1]):
        peak = peaks[index]
        directions = peak.reshape(5, 3)

        #print directions

        for i in xrange(directions.shape[0]):

            if np.linalg.norm(directions[i]) != 0:

                line_actor = fvtk.line(np.array(index) + scale*np.vstack((-directions[i], directions[i])), directions[i]/np.linalg.norm(directions[i]))
                fvtk.add(r, line_actor)
                

    fvtk.show(r)

crossings(peaks)
