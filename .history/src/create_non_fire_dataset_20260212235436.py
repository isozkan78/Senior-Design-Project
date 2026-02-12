import ee

# 1. GEE Başlatma
print("🔄 Google Earth Engine bağlantısı...")
try:
    ee.Initialize(project='fire-risk-academic')
    print("✅ Bağlantı Başarılı!")
except:
    ee.Authenticate()
    ee.Initialize(project='fire-risk-academic')

# 2. Ayarlar
roi = ee.Geometry.Rectangle([29.2, 36.0, 32.5, 37.5]) 
POINT_COUNT = 500  

print(f"📍 Rastgele {POINT_COUNT} adet 'Yangın Olmayan' nokta üretiliyor...")

# 3. Rastgele Noktalar ve Tarihler Oluşturma
# randomPoints ile noktaları oluştur, sonra her birine bir 'random' sütunu ekle
points = ee.FeatureCollection.randomPoints(roi, POINT_COUNT)
points = points.randomColumn('random_val')

def add_data(feature):
    # Rastgele bir tarih belirle (2021 Temmuz-Ağustos)
    # 1625097600000 = 1 Temmuz 2021
    # 5270400000 = ~2 aylık milisaniye
    rand_val = ee.Number(feature.get('random_val'))
    random_time = ee.Number(1625097600000).add(rand_val.multiply(5270400000).toInt())
    date = ee.Date(random_time)
    
    # Uydu verilerini çek (LST, NDVI, Elevation)
    lst = ee.ImageCollection('MODIS/006/MOD11A2') \
        .filterDate(date.advance(-10, 'day'), date.advance(2, 'day')) \
        .mean().select(['LST_Day_1km'], ['LST'])
        
    ndvi = ee.ImageCollection('MODIS/006/MOD13A1') \
        .filterDate(date.advance(-16, 'day'), date.advance(2, 'day')) \
        .mean().select(['NDVI'])
        
    srtm = ee.Image('USGS/SRTMGL1_003').select(['elevation'])
    
    full_img = lst.addBands(ndvi).addBands(srtm)
    
    stats = full_img.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=feature.geometry(),
        scale=1000
    )
    
    return feature.set(stats).set('label', 0).set('ACQ_DATE', random_time)

# 4. İşlemi Başlat
print("🛰️ Uydu görüntüleri işleniyor (Label=0)...")
dataset_final = points.map(add_data).filter(ee.Filter.notNull(['LST', 'NDVI', 'elevation']))

# 5. Drive'a Gönder
print("🚀 Google Drive'a aktarma görevi başlatılıyor...")
task = ee.batch.Export.table.toDrive(
    collection=dataset_final,
    description='Antalya_NonFire_Verisi_Final',
    fileFormat='CSV',
    selectors=['label', 'LST', 'NDVI', 'elevation', 'ACQ_DATE']
)

task.start()

print("\n✅ GÖREV BAŞARIYLA GÖNDERİLDİ!")
print("Task Manager: https://code.earthengine.google.com/tasks")