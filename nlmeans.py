import nibabel as nib
import numpy as np
import os
import tempfile


def nlmeans(img_full, cores=None, std=0, nbhood_size=10, search_size=50):
    """
    Converts an image to the .ima format as float32 for NLMEANS
    and converts it back to an int16 nifti file.
    You will also need the NLMEANS program residing in your path or in the
    same folder as your image. And don't forget to mark it as executable
    beforehand (with chmod +x NLMEANS)
    """

    # Check if the user didn't supply the optionnals parameters
    if cores is None:
        import multiprocessing
        cores = multiprocessing.cpu_count()

    # Random string generator for temp files
    import string
    import random
    length = 20
    filename = '/' + "".join([random.choice(string.letters+string.digits) for x in range(1, length)])

    # Process image and save its attributes for later
    img = img_full.get_data().astype('float32')  # NLMeans requires float
    hdr = img_full.get_header()
    dim = hdr['dim'][1:5]
    affine = img_full.get_affine()

    # Save file as .ima to use it with NLMeans
    tempdir = tempfile.gettempdir()
    write_img = open(tempdir + filename + '.ima', 'w')
    write_img.write(img)
    write_img.close()

    # Save header as .dim to use it with NLMeans. It requires the dimensions
    # as well as the data type (forced as float here) to minimally function.
    write_hdr = open(tempdir + filename + '.dim', 'w')
    write_hdr.write(str(dim).strip('[]') + '\n-type FLOAT')
    write_hdr.close()

    # Call NLMeans with the newly written file
    os.environ['PATH'] += ':./'
    os.system('export VISTAL_SMP=' + str(cores) + ' && NLMEANS -in '
              + tempdir + filename + ' -out ' + tempdir + filename + '_denoised'
              + ' -sigma ' + str(std) + ' -averaging 1' + ' -v ' + str(nbhood_size) + ' -d ' + str(search_size))

    # Read back the created file
    read_img = open(tempdir + filename + '_denoised.ima', 'rb')
    data_img = np.fromfile(tempdir + filename + '_denoised.ima').view(dtype='float32').reshape(dim[::-1]).squeeze()
    read_img.close()

    # We delete the temporary .ima/.dim files
    os.remove(tempdir + filename + '.ima')
    os.remove(tempdir + filename + '.dim')
    os.remove(tempdir + filename + '_denoised.ima')
    os.remove(tempdir + filename + '_denoised.dim')

    # Some swap magic required by the way we read/write .ima files
    # The .ima format is saved as TZYX by NLMEANS, while the previously
    # outputted image is XYZT, so we swap X and Z
    swapped_img = np.swapaxes(data_img, 0, -1)

    # A 4D image requires swapping Y and Z, while a 3D one is okay with the
    # fix above since we switched X and Z already.
    if hdr['dim'][0] == 4:
        swapped_img = np.swapaxes(swapped_img, 1, 2)

    # Convert the swapped nifti as int16 to save space, since float32
    # is bigger and probably doesn't give more noticeable precision
    img_denoised = nib.Nifti1Image(swapped_img.astype(np.int16), affine)
    return img_denoised
