import ee

def lookup(source_hist, target_hist):
    """Creates a lookup table for matching a source histogram to a target histogram."""
    source_values = source_hist.slice(1, 0, 1).project([0])
    source_counts = source_hist.slice(1, 1, 2).project([0])
    source_counts = source_counts.divide(source_counts.get([-1]))

    target_values = target_hist.slice(1, 0, 1).project([0])
    target_counts = target_hist.slice(1, 1, 2).project([0])
    target_counts = target_counts.divide(target_counts.get([-1]))

    def make_lookup(n):
            return target_values.get(target_counts.gte(n).argmax())

    return {'x': source_values.toList(), 'y': source_counts.toList().map(make_lookup)}

def find_closest(target_image, image_col, days):
    """Filter images in a collection by date proximity and spatial intersection to a target image.

    Args:
        target_image: An ee.Image whose observation date is used to find near-date images in
          the provided image_col image collection. It must have a 'system:time_start' property.
        image_col: An ee.ImageCollection to filter by date proximity and spatial intersection
          to the target_image. Each image in the collection must have a 'system:time_start'
          property.
        days: A number that defines the maximum number of days difference allowed between
          the target_image and images in the image_col.

    Returns:
        An ee.ImageCollection that has been filtered to include those images that are within the
          given date proximity to target_image and intersect it spatially.
    """

    # Compute the timespan for N days (in milliseconds).
    range = ee.Number(days).multiply(1000 * 60 * 60 * 24)

    filter = ee.Filter.And(
        ee.Filter.maxDifference(range, 'system:time_start', None, 'system:time_start'),
        ee.Filter.intersects('.geo', None, '.geo'))

    closest = (ee.Join.saveAll('matches', 'measure')
        .apply(ee.ImageCollection([target_image]), image_col, filter))

    return ee.ImageCollection(ee.List(closest.first().get('matches')))

# def match_histograms(source_img, target_col, geometry, days, bands):
#     """Matches histograms of selected bands of a source image to the closest matching target image in target_col.

#     Args:
#         source_img (ee.Image): A single image from the source collection for histogram matching.
#         target_col (ee.ImageCollection): Collection of target images for reference histograms.
#         geometry (ee.Geometry): Region for histogram calculation.
#         days (int): Maximum allowed date proximity (in days) between source_img and target images.
#         bands (list): List of band names to apply histogram matching.

#     Returns:
#         ee.Image: A color-matched image with histograms matched to the closest available target image.
#     """
#     target_image = find_closest(source_img, target_col, days).sort('CLOUD_COVER').mosaic()
    
#     args = {
#         'reducer': ee.Reducer.autoHistogram(maxBuckets=256, cumulative=True),
#         'geometry': geometry,
#         'scale': 1,
#         'maxPixels': 65536 * 4 - 1,
#         'bestEffort': True
#     }
    
#     # Compute histograms for selected bands
#     source_hist = source_img.reduceRegion(**args)
#     target_hist = target_image.updateMask(source_img.select(0).mask()).reduceRegion(**args)
#     #target_hist = target_image.updateMask(source_img.mask().select(target_image.bandNames())).reduceRegion(**args)

    
#     # Apply histogram matching only to selected bands
#     matched_bands = []
#     for band in bands:
#         matched_band = source_img.select([band]).interpolate(
#             **lookup(source_hist.getArray(band), target_hist.getArray(band))
#         )
#         matched_bands.append(matched_band)
    
#     # Combine matched bands and preserve properties
#     matched_image = ee.Image.cat(matched_bands).copyProperties(source_img, ['system:time_start'])
    
#     return matched_image

def match_histograms(source_img, target_col, geometry, days):
    """Matches histograms of each source image in a collection to the closest matching target image in target_col.

    Args:
        source_img: A single ee.Image from the source collection for which histogram matching will be performed.
        target_col: An ee.ImageCollection of target images to find the closest match for histogram reference.
        geometry: An ee.Geometry to define the region for histogram calculation.
        days: The maximum number of days allowed for date proximity between source_img and the target images.

    Returns:
        A color-matched ee.Image with histograms matched to the closest available target image.
    """
    target_image = find_closest(source_img, target_col, days).sort('CLOUD_COVER').mosaic()
    args = {
        'reducer': ee.Reducer.autoHistogram(maxBuckets=256, cumulative=True),
        'geometry': geometry,
        'scale': 1,
        'maxPixels': 65536 * 4 - 1,
        'bestEffort': True
    }
    # Reduce regions for source and target histograms
    source_hist = source_img.reduceRegion(**args)
    target_hist = target_image.updateMask(source_img.mask()).reduceRegion(**args)

    # Apply histogram matching to each of the six bands
    matched_image = ee.Image.cat(
        source_img.select(['red']).interpolate(**lookup(source_hist.getArray('red'), target_hist.getArray('red'))),
        source_img.select(['green']).interpolate(**lookup(source_hist.getArray('green'), target_hist.getArray('green'))),
        source_img.select(['blue']).interpolate(**lookup(source_hist.getArray('blue'), target_hist.getArray('blue'))),
        source_img.select(['nir']).interpolate(**lookup(source_hist.getArray('nir'), target_hist.getArray('nir'))),
        source_img.select(['swir1']).interpolate(**lookup(source_hist.getArray('swir1'), target_hist.getArray('swir1'))),
        source_img.select(['swir2']).interpolate(**lookup(source_hist.getArray('swir2'), target_hist.getArray('swir2'))),
        source_img.select(['ndvi']).interpolate(**lookup(source_hist.getArray('ndvi'), target_hist.getArray('ndvi')))
    ).copyProperties(source_img, ['system:time_start'])

    return matched_image

def prep_landsat(image):
    # only for historgram matching
    """Scale, apply cloud/shadow mask, and select/rename Landsat 8 bands."""
    qa_mask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)

    def get_factor_img(factor_names):
        factor_list = image.toDictionary().select(factor_names).values()
        return ee.Image.constant(factor_list)

    scale_img = get_factor_img(['REFLECTANCE_MULT_BAND_.'])
    offset_img = get_factor_img(['REFLECTANCE_ADD_BAND_.'])
    scaled = image.select('SR_B.').multiply(scale_img).add(offset_img)

    return image.addBands(scaled, None, True).select(
        ['SR_B4', 'SR_B3', 'SR_B2', 'SR_B5', 'SR_B6', 'SR_B7'], ['red', 'green', 'blue', 'nir', 'swir1', 'swir2']).updateMask(qa_mask)