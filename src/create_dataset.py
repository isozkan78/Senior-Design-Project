import ee

# 1. GEE Başlatma
print("🔄 Google Earth Engine bağlantısı kontrol ediliyor...")
try:
    ee.Initialize(project='fire-risk-academic') 
    print("✅ Bağlantı Başarılı!")
except Exception as e:
    print("⚠️ Yetki yenileniyor...")
    ee.Authenticate()
    ee.Initialize(project='fire-risk-academic')

# 2. Ayarlar (Antalya Bölgesi ve Tarih)
roi = ee.Geometry.Rectangle([29.2, 36.0, 32.5, 37.5]) 
start_date = '2021-07-01' # Yangınların yoğun olduğu yaz dönemi
end_date = '2021-08-30' 

print(f"📍 Bölge: Antalya | Tarih: {start_date} - {end_date}")

# 3. Veri Çekme Fonksiyonu
def get_features(feature):
    # Tarihi özelliklerden geri oku
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

# 4. Veri Setini Oluşturma (BUG FIX BURADA)
print("🔥 Yangın verileri işleniyor...")

fire_collection = ee.ImageCollection('FIRMS') \
    .filterDate(start_date, end_date) \
    .filterBounds(roi)

def process_fire_image(img):
    # Görüntünün tarihini al
    img_date = img.date()
    
    # Vektöre çevir (toInt ve select(0) hatayı önler)
    vectors = img.select(0).toInt().reduceToVectors(
        geometry=roi, 
        scale=1000, 
        geometryType='centroid', 
        labelProperty='label'
    )
    
    # KİLİT NOKTA: Tarihi (ACQ_DATE) her bir noktaya elle ekle
    return vectors.map(lambda f: f.set('ACQ_DATE', img_date.millis()))

# map() ile her görüntüye uygula ve flatten() ile tek listeye indir
dataset = fire_collection.map(process_fire_image).flatten()

# Etiketle: 1 = Yangın
dataset = dataset.map(lambda f: f.set('label', 1))

# 5. Uydu Verilerini Eşle
print("🛰️ Uydu görüntüleri (Sıcaklık, NDVI, Yükseklik) eşleştiriliyor...")
dataset_processed = dataset.map(get_features)

# Boş verileri temizle
dataset_final = dataset_processed.filter(ee.Filter.notNull(['LST', 'NDVI', 'elevation']))

# 6. Drive'a Gönder
print("🚀 Google Drive'a aktarma görevi başlatılıyor...")

task = ee.batch.Export.table.toDrive(
    collection=dataset_final,
    description='Antalya_Yangin_Verisi_Tam',
    fileFormat='CSV',
    selectors=['label', 'LST', 'NDVI', 'elevation', 'ACQ_DATE']
)

task.start()

print("\n✅ GÖREV BAŞARIYLA GÖNDERİLDİ!")
print("Task Manager: https://code.earthengine.google.com/tasks")