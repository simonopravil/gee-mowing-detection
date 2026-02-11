import ee

def apply_cs_plus_mask(image_collection, maskBands, maskConf):
    """
    Apply Cloud Score Plus masking to an image collection.
    
    Args:
        image_collection (ee.ImageCollection): Input image collection
        params (dict): Dictionary containing:
            - QA_BAND: Quality assurance band name (e.g., 'cs_plus')
            - maskConf: Clean score threshold (0-100)
    
    Returns:
        ee.ImageCollection: Image collection with Cloud Score Plus mask applied
    """
    def _applyMask(img):
        return (img
        .updateMask(img.select(maskBands).gte(maskConf / 100))
        
        .addBands(img.select(maskBands).gte(maskConf / 100).rename('mask'))
        )

                
    # Get Cloud Score Plus collection
    csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
    
    # Link the Cloud Score Plus collection and apply masking
    masked_collection = (image_collection
        .linkCollection(csPlus, [maskBands])
        .map(_applyMask)
    )
    
    return masked_collection

def compute_cloud_cover(ROI):
    def _compute_cloud_cover(image):
        # Cloud probability band in Sentinel-2 images.
        cloud_mask = image.select('mask')  # Scene classification: 9 represents "clouds".
        
        # Calculate cloud cover percentage over the ROI.
        cloud_cover = cloud_mask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ROI, #Resolve
            scale=10,
            maxPixels=1e9
        ).get('mask')
    
    # Add the calculated cloud cover as a property to the image.
        return image.set('cloud_cover_roi', cloud_cover)
    return _compute_cloud_cover


def mask_landsat_with_conf(image_collection, conf = 'Medium'):
    """
    Apply a Landsat mask to an image collection, considering cloud, cloud shadow, snow,
    fill, dilated clouds, and confidence levels. Additionally, adds a Cloud Score band
    using `simpleCloudScore()` and computes CleanScore for each image.

    Args:
        image_collection (ee.ImageCollection): Input image collection
        conf (str): Confidence level ('Low', 'Medium', 'High')

    Returns:
        ee.ImageCollection: Image collection with mask applied and Cloud Score band (`mask_conf`).
    """
    
    # Define mask bits for cloud, cloud shadow, snow, fill, dilated
    dict_mask = {
        'cloud': ee.Number(2).pow(3).int(),
        'cshadow': ee.Number(2).pow(4).int(),
        'snow': ee.Number(2).pow(5).int(),
        'fill': ee.Number(2).pow(0).int(),
        'dilated': ee.Number(2).pow(1).int(),
    }

    # Define confidence thresholds
    dict_conf = {'Low': 1, 'Medium': 2, 'High': 3}
    sel_conf = ee.Number(dict_conf[conf])

    # Combine all relevant mask bits (cloud, cloud shadow, snow, fill, dilated)
    all_bits = ee.Number(0)
    for mask in dict_mask.values():
        all_bits = all_bits.add(mask)
    
    def extractQAbits(qa_band, bit_start, bit_end):
        """
        Helper function to extract the QA bits from a Landsat image.
        """
        numbits = ee.Number(bit_end).subtract(ee.Number(bit_start)).add(ee.Number(1))
        qa_bits = qa_band.rightShift(bit_start).mod(ee.Number(2).pow(numbits).int())
        return qa_bits

    def _applyMask(img):
        # Define the QA band for Landsat (QA_PIXEL)
        qa = img.select('QA_PIXEL')
        mask = qa.bitwiseAnd(all_bits).eq(0)  # Exclude fill, dilated cloud, cirrus, cloud, and cloud shadow

        # Extract cloud confidence (bits 8-9) and apply based on user-defined confidence level
        cloud_conf = extractQAbits(qa, 8, 9)
        cloud_low = cloud_conf.gte(sel_conf).Not()  # Exclude high-confidence clouds
        mask = mask.And(cloud_low)

        # Extract cirrus confidence (bits 14-15) and apply based on user-defined confidence level
        cirr_conf = extractQAbits(qa, 14, 15)
        cirr_low = cirr_conf.gte(sel_conf).Not()  # Exclude high-confidence cirrus
        mask = mask.And(cirr_low)

        # QA_RADSAT (radiometric saturation) check - should be equal to 0 (not saturated)
        saturation = img.select('QA_RADSAT').eq(0)
        mask = mask.And(saturation)

        # Valid minima and maxima checks for the main bands (should be between 0 and 1)
        valid_min = img.select(['blue', 'green', 'red','nir','swir1', 'swir2']).reduce(ee.Reducer.min()).gt(0)
        valid_max = img.select(['blue', 'green', 'red','nir','swir1', 'swir2']).reduce(ee.Reducer.max()).lt(1)
        mask = mask.And(valid_min).And(valid_max)

        # Final mask applied to the image
        mask = mask.rename('mask')
        return img.updateMask(mask).addBands(mask).copyProperties(source=img).set('system:time_start', img.get('system:time_start'))

    # Apply the mask to each image in the collection
    masked_collection = image_collection.map(_applyMask)

    return masked_collection


