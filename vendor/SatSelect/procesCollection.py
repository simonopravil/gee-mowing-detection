import ee

def rename_bands_l8(img):
    originalNames = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL', 'QA_RADSAT']
    commonNames = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'QA_PIXEL', 'QA_RADSAT']
    return img.select(originalNames).rename(commonNames).set('satellite', 'L8')

def rename_bands_l9(img):
    originalNames = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'QA_PIXEL', 'QA_RADSAT']
    commonNames = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'QA_PIXEL', 'QA_RADSAT']
    return img.select(originalNames).rename(commonNames).set('satellite', 'L9')

def rename_bands_s2(img):
    originalNames = ['B2','B3','B4','B5', 'B6', 'B7', 'B8','B11','B12']
    commonNames = ['blue', 'green', 'red','re1', 're2', 're3', 'nir', 'swir1', 'swir2']
    return img.select(originalNames).rename(commonNames).set('satellite', 'S2')

def scale_bands(bands, scale=1e4, offset=0.0):
    def wrap(img):
        imgs = img.select(bands).multiply(scale).add(offset)
        return img.addBands(imgs, overwrite=True)
    return wrap

def renameCloudCoverS2(img):
    # For use in Mosaic algorithm.
    return img.set('CLOUD_COVER', img.get('CLOUDY_PIXEL_PERCENTAGE'))

def satellite_order_S2(img):
    # For use in Mosaic to select Sentinel over Landsat
    return img.set('order', 1)

def satellite_order_L(img):
    # For use in Mosaic to select Sentinel over Landsat
    return img.set('order', 2)

def sentinel2_preproc(aoi, startDate, endDate, tileCldFilter):
    filter = ee.Filter.And(
        ee.Filter.bounds(aoi),
        ee.Filter.date(startDate, endDate)
    )
    
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filter(filter)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', tileCldFilter))
          .map(rename_bands_s2)
          .map(scale_bands(['blue', 'green', 'red', 'nir', 'swir1', 'swir2'], scale=0.0001, offset=0))
          .map(satellite_order_S2)
          )

    return s2

def landsat_preproc(aoi, startDate, endDate, tileCldFilter):
    filter = ee.Filter.And(
        ee.Filter.bounds(aoi),
        ee.Filter.date(startDate, endDate)
    )

    l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .filter(filter)
          .filter(ee.Filter.lte('CLOUD_COVER_LAND', tileCldFilter))
          .map(rename_bands_l8)
          .map(scale_bands(
              ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'], 
              scale=0.0000275, offset=-0.2)
          )
          .map(satellite_order_L)
    )
    
    l9 = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
          .filter(filter)
          .filter(ee.Filter.lte('CLOUD_COVER_LAND', tileCldFilter))
          .map(rename_bands_l9)
          .map(scale_bands(
              ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'], 
              scale=0.0000275, offset=-0.2)
          )
          .map(satellite_order_L)
    )  

    landsat = l8.merge(l9).sort("system:time_start") 
    return landsat
