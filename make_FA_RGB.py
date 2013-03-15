import numpy as np
from dipy.reconst.dti import fractional_anisotropy, color_fa, TensorModel


def FA_RGB(data, gtab):
    """
    Input : data, gtab taken from the load_data.py script.
    Return : FA and RGB as two nd numpy array
    """

    tenmodel = TensorModel(gtab)
    tenfit = tenmodel.fit(data)
    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)] = 0
    FA = np.clip(FA, 0, 1)
    RGB = color_fa(FA, tenfit.evecs)
    return FA, RGB