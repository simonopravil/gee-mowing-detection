import ee

def MosaicByDate(originalCollection):
    
    def unique_values(collection, field):
        values = ee.Dictionary(collection.reduceColumns(ee.Reducer.frequencyHistogram(), [field]).get('histogram')).keys()
        return values

    def daily_mosaics(imgs):

        def simplifyDate(img):
            d = ee.Date(img.get('system:time_start'))
            day = d.get('day')
            m = d.get('month')
            y = d.get('year')
            simpleDate = ee.Date.fromYMD(y,m,day)
            return img.set('simpleTime',simpleDate.millis())

        imgs = imgs.map(simplifyDate)
        days = unique_values(imgs,'simpleTime')

        def do_mosaic(d):
            d = ee.Number.parse(d)
            d = ee.Date(d)
            t = imgs.filterDate(d,d.advance(1,'day')).sort('order').sort('CLOUD_COVER')
            f = ee.Image(t.first())
            t = t.mosaic()
            t = t.set('system:time_start',d.millis())
            t = t.copyProperties(f)
            return t

        imgs = days.map(do_mosaic)
        
        return ee.ImageCollection.fromImages(imgs)
    
    mosaiked = daily_mosaics(originalCollection)
    return mosaiked

def imgcol_dates_to_featcol(imgcol, format='YYYYMMdd'):
    dates = imgcol.map(
        lambda img: ee.Feature(None, {'system:time_start': img.date().millis(),
                                      'YYYYMMDD': img.date().format(format)})
    )
    return ee.FeatureCollection(dates)
    
def days_to_milli(days):
    return days*1000*60*60*24

def mosaic_imgcol(imgcol):
    unique_dates = imgcol_dates_to_featcol(imgcol).distinct('YYYYMMDD')  # unique YYYYMMDD
    newcol = ee.ImageCollection(
        ee.Join.saveAll('images').apply(
            primary=unique_dates, secondary=imgcol,
            condition=ee.Filter.maxDifference(  
                difference=days_to_milli(0.5),
                leftField='system:time_start', rightField='system:time_start'
            )
        )
    )
    mosaics = newcol.map(
        lambda x: ee.ImageCollection(ee.List(x.get('images'))).mosaic().set('system:time_start', x.get('system:time_start'))
    ).sort("system:time_start")
    return mosaics