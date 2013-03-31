import nibabel as nib
from dipy.reconst.shm import sh_to_sf


def show_odf_sample(filename):

    odf_sh = nib.load(filename).get_data()

    from dipy.data import get_sphere

    sphere = get_sphere('symmetric724')

    odf = sh_to_sf(odf_sh, sphere, 8, 'mrtrix')

    from dipy.viz import fvtk
    r = fvtk.ren()
    #odf = odf[:, :, 25]
    #fvtk.add(r, fvtk.sphere_funcs(odf[:, :, None], sphere, norm=True))
    odf = odf[:, 22, :]
    #odf = odf[14: 23, 22, 34: 43]
    #odf = odf[14:24, 22, 23:33]
    fvtk.add(r, fvtk.sphere_funcs(odf[:, None, :], sphere, norm=True))

    fvtk.show(r)

    #return odf[25 - 10:25 + 10, 25 - 10:25 + 10, 25]

if __name__ == '__main__':
	
	import sys

	show_odf_sample(sys.argv[1])
