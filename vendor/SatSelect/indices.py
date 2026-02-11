import ee
def ndvi(img):
       ndvi = img.normalizedDifference(['nir', 'red']).rename('ndvi')
       return img.addBands(ndvi)

# kernalized Normalized Difference Vegetation Index (kNDVI); Camps-Valls et al. 2021
# https://www.science.org/doi/10.1126/sciadv.abc7447
def kndvi(img):
     kndvi = ee.Image(
          ((img.normalizedDifference(['nir', 'red'])).pow(2)).tanh()
     ).rename('kndvi')
     return img.addBands(kndvi)

# Enhanced Vegetation Index (EVI); Huete et al. (2002)
def evi(gain=2.5, l=1, c1=6, c2=7.5):
    def wrap(img):
        evi = img.expression(
            'gain * ((nir - red) / (nir + c1 * red - c2 * blue + l))',
            {
                'gain': gain,
                'nir': img.select('nir'),
                'red': img.select('red'),
                'blue': img.select('blue'),
                'c1': c1,
                'c2': c2,
                'l': l
            }
        ).rename('evi')
        return img.addBands(evi)
    return wrap

def apply_indices(collection, indices):
    """Apply selected indices to an image collection."""
    if 'ndvi' in indices:
        collection = collection.map(ndvi)
    if 'evi' in indices:
        collection = collection.map(evi())
    if 'kndvi' in indices:
        collection = collection.map(kndvi)
    return collection