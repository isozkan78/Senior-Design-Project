import ee
import random

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
start_date = '2021-07-01'
end_date = '2021-08-30'
POINT_COUNT = 500  # Kaç tane 'Yangın Yok' verisi üretelim?

print(f"📍 Rastgele {POINT_COUNT} adet 'Yangın Olmayan' nokta üretiliyor...")

# 3. Yardımcı Fonksiyon: Rastgele Tarih Üretme
# GEE üzerinde rastgele tarih atamak zor olduğu için, noktaları oluştururken
# her birine Python tarafında rastgele bir milisaniye (zaman damgası) vereceğiz.
def add_random_date(feature):
    # 2021 Temmuz-Ağustos arası rastgele zaman (Unix Timestamp)
    # 1625097600000 (1 Temmuz) - 1630368000000 (31 Ağustos)
    random_time = ee.Number(1625097600000).add(ee.Number(random.randint(0, 5270400000)))
    return feature.set('ACQ_DATE', random_time)

# 4. Veri Çekme Fonksiyonu (Öncekinin Aynısı)
def get_features(feature):
    date = ee.Date(feature.get('ACQ_DATE'))
    
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
    return feature.set(stats)

# 5. Rastgele Noktaları Oluştur
# Rastgele 500 nokta üret
random_points = ee.FeatureCollection.randomPoints(roi, POINT_COUNT)

# Her noktaya rastgele bir tarih ata (Server-side map içinde random kullanamayız, basit trick yapıyoruz)
# Burada basitlik adına: GEE'ye rastgele sayı ürettirip tarihe ekliyoruz.
random_points = random_points.map(lambda f: f.set('ACQ_DATE', 
                                                  ee.Number(1625097600000).add(ee.Number.random().multiply(5000000000).toInt())
                                                 ))

# Etiketle: 0 = Yangın Yok
dataset = random_points.map(lambda f: f.set('label', 0))

# 6. Uydu Verilerini Eşle
print("🛰️ Uydu görüntüleri işleniyor (Label=0)...")
dataset_processed = dataset.map(get_features)

# Boş verileri temizle
dataset_final = dataset_processed.filter(ee.Filter.notNull(['LST', 'NDVI', 'elevation']))

# 7. Drive'a Gönder
print("🚀 Google Drive'a aktarma görevi başlatılıyor...")

task = ee.batch.Export.table.toDrive(
    collection=dataset_final,
    description='Antalya_NonFire_Verisi',
    fileFormat='CSV',
    selectors=['label', 'LST', 'NDVI', 'elevation', 'ACQ_DATE']
)

task.start()

print("\n✅ 'YANGIN YOK' GÖREVİ BAŞLATILDI!")
print("Task Manager: https://code.earthengine.google.com/tasks")