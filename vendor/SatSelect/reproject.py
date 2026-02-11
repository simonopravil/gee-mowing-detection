import ee

def repr(IC, crs, scale):
    def _reproject(img):
        return img.reproject(crs = crs, scale = scale)
    return IC.map(_reproject)


def res(IC, method = 'nearest-neighbor'):
    # default Nearest neighbor
    def _resample(img):
        return img.resample(method)
    return IC.map(_resample)