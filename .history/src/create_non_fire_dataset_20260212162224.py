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

# 3. Veri Çekme Fonksiyonu
def get_features(feature):
    # Her noktaya rastgele bir tarih ata (2021 Temmuz-Ağustos arası)
    # 1625097600000 (1 Temmuz) + rastgele milisaniye
    rand_num = ee.Number(ee.RandomColumn(ee.FeatureCollection([feature])).first().get('random'))
    random_time = ee.Number(1625097600000).add(rand_num.multiply(5184000000).toInt())
    
    date = ee.Date(random_time)
    
    # a) Sıcaklık (LST)
    lst = ee.ImageCollection('MODIS/006/MOD11A2') \
        .filterDate(date.advance(-10, 'day'), date.advance(2, 'day')) \
        .mean().select(['LST_Day_1km'], ['LST'])
        
    # b) Bitki Örtüsü (NDVI)
    ndvi = ee.ImageCollection('MODIS/006/MOD13A1') \
        .filterDate(date.advance(-16, 'day'), date.advance(2, 'day')) \
        .mean().select(['NDVI'])
        
    # c) Yükseklik - SRTM
    srtm = ee.Image('USGS/SRTMGL1_003').select(['elevation'])
    
    full_img = lst.addBands(ndvi).addBands(srtm)
    
    stats = full_img.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=feature.geometry(),
        scale=1000
    )
    return feature.set(stats).set('label', 0).set('ACQ_DATE', random_time)

# 4. Rastgele Noktaları Oluştur ve İşle
# randomPoints içine toInt() hatasını almamak için doğrudan FeatureCollection kullanıyoruz
random_points = ee.FeatureCollection.randomPoints(roi, POINT_COUNT)

print("🛰️ Uydu görüntüleri işleniyor (Label=0)...")
dataset_processed = random_points.map(get_features)

# Boş verileri temizle
dataset_final = dataset_processed.filter(ee.Filter.notNull(['LST', 'NDVI', 'elevation']))

# 5. Drive'a Gönder
print("🚀 Google Drive'a aktarma görevi başlatılıyor...")

task = ee.batch.Export.table.toDrive(
    collection=dataset_final,
    description='Antalya_NonFire_Verisi_Fixed',
    fileFormat='CSV',
    selectors=['label', 'LST', 'NDVI', 'elevation', 'ACQ_DATE']
)

task.start()

print("\n✅ GÖREV BAŞARIYLA GÖNDERİLDİ!")
print("Task Manager: https://code.earthengine.google.com/tasks")