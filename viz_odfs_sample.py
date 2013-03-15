import nibabel as nib


def show_odf_sample(filename):

    odf = nib.load(filename).get_data()

    from dipy.data import get_sphere

    sphere = get_sphere('symmetric724')
    
    from dipy.viz import fvtk
    r = fvtk.ren()
    # fvtk.add(r, fvtk.sphere_funcs(odf[:, :, 25], sphere))
    fvtk.add(r, fvtk.sphere_funcs(odf[25 - 10:25 + 10, 25 - 10:25 + 10, 25], sphere))
    fvtk.show(r)

if __name__ == '__main__':
	
	import sys

	show_odf_sample(sys.argv[1])
