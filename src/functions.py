import ee 


def create_grid(geometry, crs, pixelSize = 10, tileSize = 10000):
    # Choose the export CRS
    crs = crs
    
    # Choose the pixel size for export (meters)
    pixelSize = pixelSize
    
    # Choose the export tile size (pixels)
    tileSize = tileSize
    
    # Calculate the grid size (meters)
    gridSize = tileSize * pixelSize
    
    # Create the grid covering the geometry bounds
    bounds = geometry.bounds(**{
    'proj': crs, 'maxError': 1
    })
    
    grid = bounds.coveringGrid(**{
    'proj':crs, 'scale': gridSize
    })

    # Calculate the coordinates of the top-left corner of the grid
    bounds = grid.geometry().bounds(**{
    'proj': crs, 'maxError': 1
    });
    
    # Extract the coordinates of the grid
    coordList = ee.Array.cat(bounds.coordinates(), 1)
    
    xCoords = coordList.slice(1, 0, 1)
    yCoords = coordList.slice(1, 1, 2)
    
    # We need the coordinates of the top-left pixel
    xMin = xCoords.reduce('min', [0]).get([0,0])
    yMax = yCoords.reduce('max', [0]).get([0,0])
    
    # Create the CRS Transform
    
    # The transform consists of 6 parameters:
    # [xScale, xShearing, xTranslation, 
    #  yShearing, yScale, yTranslation]
    transform = ee.List([
        pixelSize, 0, xMin, 0, -pixelSize, yMax]).getInfo()

    return grid, bounds, transform

TIME_AXIS = 0
BAND_AXIS = 1

def get_shift(array, shift_type):
    time = array.arraySlice(BAND_AXIS, -1)
    sorted_array = array.arraySort(time)
    padding = ee.Image(0).toArray().arrayRepeat(BAND_AXIS, 1)
    double_padding = padding.arrayCat(padding, TIME_AXIS)
    length = sorted_array.arrayLength(TIME_AXIS)
    ts = sorted_array.arraySlice(BAND_AXIS, 0, 1)

    if shift_type == 'm1':
        ts_shift = padding.arrayCat(ts.arraySlice(TIME_AXIS, 0, -1), TIME_AXIS)
    elif shift_type == 'p1':
        ts_shift = ts.arraySlice(TIME_AXIS, 1).arrayCat(padding, TIME_AXIS)
    elif shift_type == 'm2':
        ts_shift = double_padding.arrayCat(ts.arraySlice(TIME_AXIS, 0, -2), TIME_AXIS)
    elif shift_type == 'p2':
        ts_shift = ts.arraySlice(TIME_AXIS, 2).arrayCat(double_padding, TIME_AXIS)
    else:
        raise ValueError('Invalid shift_type')

    return ts_shift.arraySlice(TIME_AXIS, 0, length)

def CreateDiffColl(array, prefix, shift_type = 'm1'):

    def compute(image):
        arrayTime = array.arraySlice(BAND_AXIS, -1)
        timeMask = arrayTime.eq(image.getNumber('system:time_start')).arrayProject([TIME_AXIS])
        shifted = get_shift(array, shift_type).arrayProject([TIME_AXIS]).arrayMask(timeMask)
        shifted = shifted.updateMask(shifted.arrayLength(TIME_AXIS)).arrayFlatten([[prefix + shift_type]])
        return image.addBands(shifted)

    return compute

def reduce_quantile(ic, grass_mask, tiles, quantile, crs, scale):
    
    def reduce_region(img):

        #img = img#.updateMask(grass_mask).select('ndvi')
        # Create the reducer
        stats = img.select('ndvi').updateMask(grass_mask).reduceRegions(
            collection = tiles,
            reducer = ee.Reducer.percentile([quantile]).setOutputs(['spatial_max']),
            scale = scale,
            crs = crs,
            tileScale = 8
        )
        def fill_missing(f):
            p = ee.Number(f.get('spatial_max'))
            # Test explicitly for None (masked result)
            p_filled = ee.Algorithms.If(
                ee.Algorithms.IsEqual(p, None),
                -9999,  # Default value for masked areas
                p
            )
            return f.set({'spatial_max': p_filled})
        
        stats = stats.map(fill_missing)
        # Apply the conversion
        grid_image = stats.reduceToImage(
            properties=['spatial_max'],
            reducer=ee.Reducer.first()
        ).rename('spatial_max')

        return img.addBands(grid_image)
    
    return ic.map(reduce_region)

def addTimeBands(image):
    date = image.date()
    doy = date.getRelative('day', 'year')
    time = image.metadata('system:time_start')
    doyImage = ee.Image(doy).rename('doy').int()
    return image.addBands([doyImage, time])

def weighted_majority_filter(img):
    """
    Apply weighted majority filter (center-weighted 3x3).
    Gives more weight to center and adjacent pixels.
    """
    weights_5 = [[1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1]
               ]

    kernel = ee.Kernel.fixed(5, 5, weights_5)

    filtered = img.reduceNeighborhood(
        reducer=ee.Reducer.mode(),
        kernel=kernel
    )

    return img.addBands(filtered)

def updateCloudMask(img):
  return img.updateMask(img.select('mask'))

def maxdifference(img):
    diff = img.select('ndvi').subtract(img.select('spatial_max'))
    return img.addBands(diff.rename('diff_max'))

def time_diff(ic, band):
    def _time_diff(img):
        t0 = img.select(band)
        tm1 = img.select(f'{band}m1')
        diff = (t0.subtract(tm1)).rename(f'{band}_neg_diff')
        return img.addBands(diff)
    return ic.map(_time_diff)

def get_trained_model(bands):
    df = ee.FeatureCollection('projects/ee-simonopravil/assets/DisertationProject/mowing_predictors_toEE')
    rf = (ee.Classifier.smileRandomForest(numberOfTrees = 500,  minLeafPopulation = 4,
                                          bagFraction = 1, maxNodes = 75, seed=42)
          .train(df, classProperty = 'label', inputProperties = bands)
          .setOutputMode('MULTIPROBABILITY'))
    return rf

# def predict_model(ic, model):
#     def _predict_model(img):
#         classified = img.classify(model)
#         probabilities = (classified.arrayFlatten([PROB_BANDS[AOI_NAME]])
#                      .multiply(10000)
#                      .toUint16())
#         return img.addBands(img.classify(model).rename('mowing')).select('mowing')
#     return ic.map(_predict_model)

def predict_model(ic, model):
    def _predict_model(img):
        # Get classification with probabilities (mode='PROBABILITY')
        probabilities = img.classify(model)
        
        # Extract probability of class 1 (mowing)
        prob_mowing = probabilities.arrayGet([1]).rename('prob_mowing')
        
        # Apply 0.35 threshold to create binary classification
        mowing = prob_mowing.gte(0.35).rename('mowing')
        
        # Add both probability and binary classification bands
        return img.addBands(prob_mowing).addBands(mowing)
    
    return ic.map(_predict_model)

def filter_close_mowing_events(collection, min_interval_days=20):
    """
    Filters mowing events from an ImageCollection by removing any mowing event
    that occurs within `min_interval_days` of a previous mowing event (per pixel).

    Parameters:
    -----------
    collection : ee.ImageCollection
        The input collection containing at least 'mowing' and 'system:time_start' bands.

    min_interval_days : int
        Minimum interval (in days) between consecutive mowing events to be kept (default = 20).

    Returns:
    --------
    ee.ImageCollection
        The same collection with an added band 'mowing_filtered' per image.
    """

    min_diff_millis = min_interval_days * 24 * 60 * 60 * 1000

    primary = collection.sort('system:time_start')
    secondary = collection.sort('system:time_start')

    # Join: find earlier images within the interval
    filter = ee.Filter.And(
        ee.Filter.maxDifference(
            difference=min_diff_millis,
            leftField='system:time_start',
            rightField='system:time_start'
        ),
        ee.Filter.greaterThan(
            leftField='system:time_start',
            rightField='system:time_start'
        )
    )

    join = ee.Join.saveAll('matches')
    joined = join.apply(primary, secondary, filter)

    def process_image(img):
        img = ee.Image(img)
        mowing = img.select('mowing_mode')
        matches = ee.List(img.get('matches'))

        def extract_mowing(match):
            return ee.Image(match).select('mowing_mode')

        has_recent_mowing = ee.Algorithms.If(
            matches.size().gt(0),
            ee.ImageCollection.fromImages(matches.map(extract_mowing)).max(),
            ee.Image.constant(0).rename('mowing')
        )

        has_recent_mowing = ee.Image(has_recent_mowing)
        # Only keep mowing if no recent mowing is found
        mowing_filtered = mowing.where(has_recent_mowing.eq(1), 0)

        return img.addBands(mowing_filtered.rename('mowing_filtered'), None, True)

    return ee.ImageCollection(joined.map(process_image))

def extract_mowing_events(image):
    mowing = image.select('mowing_filtered')
    doy = image.select('doy')
    mowing_doy = doy.updateMask(mowing)
    return mowing_doy.rename('mowingEvent')

def mowing_results(ic):
    # Map to get only DOY where mowing happened
    array_image = ic.map(extract_mowing_events).toArray()
    
    # Slice to select up to 5 mowing events
    image = (array_image
             .arraySlice(1)
             .arraySlice(0,0,5)
             .arrayPad([5, 0], 0)
    )
    
    # Flatten the array to separate bands
    result = image.arrayProject([0]).arrayFlatten([
        ['mowing1DOY', 'mowing2DOY', 'mowing3DOY', 'mowing4DOY', 'mowing5DOY']
    ])
    
    return result
