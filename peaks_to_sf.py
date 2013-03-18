import numpy as np
from dipy.core.sphere import Sphere
from dipy.data import get_sphere

sphere = get_sphere('symmetric724')

peaks = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

angles = np.abs(np.dot(peaks, sphere.vertices.T))

odf = np.zeros(sphere.vertices.shape[0])

for a in angles:
    i = np.argsort(a)[-2:]
    odf[i]=1

from dipy.viz import fvtk

r=fvtk.ren()
fvtk.add(r, fvtk.sphere_funcs(odf+0.5, sphere))
#fvtk.add(r, fvtk.axes())
fvtk.show(r)


