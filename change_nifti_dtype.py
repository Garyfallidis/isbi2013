import nibabel as nib


def change_type(filename):

    img = nib.load(filename)
    data = img.get_data()
    affine = img.get_affine()

    filename2 = filename.split('.nii.gz')[0] + '_f4.nii.gz'
    print filename2

    nib.save(nib.Nifti1Image(data.astype('f4'), affine), filename2)
    

if __name__ == '__main__':
    
    import sys

    change_type(sys.argv[1])
