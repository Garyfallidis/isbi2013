import nibabel as nib


def show_odf_sample(filename):

    odf = nib.load(filename).get_data()

    from dipy.data import get_sphere

    sphere = get_sphere('symmetric724')

    from dipy.viz import fvtk
    r = fvtk.ren()
    #odf = odf[:, :, 25]
    #fvtk.add(r, fvtk.sphere_funcs(odf[:, :, None], sphere, norm=True))
    odf = odf[:, 22, :]
    fvtk.add(r, fvtk.sphere_funcs(odf[:, None, :], sphere, norm=True))

    fvtk.show(r)

    #return odf[25 - 10:25 + 10, 25 - 10:25 + 10, 25]

if __name__ == '__main__':
	
	import sys

	show_odf_sample(sys.argv[1])
