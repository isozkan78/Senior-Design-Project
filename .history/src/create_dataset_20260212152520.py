import ee

# 1. GEE Başlatma
print("🔄 Google Earth Engine bağlantısı kontrol ediliyor...")
try:
    ee.Initialize()
    print("✅ Bağlantı Başarılı!")
except Exception as e:
    print("⚠️ Yetki yenileniyor...")
    ee.Authenticate()
    ee.Initialize()

# 2. Ayarlar (Antalya Bölgesi ve Tarih)
# Antalya'yı kapsayan dikdörtgen koordinatlar
roi = ee.Geometry.Rectangle([29.2, 36.0, 32.5, 37.5]) 
start_date = '2021-01-01'
end_date = '2021-12-31' # Hızlı olması için 1 yıllık veri çekelim şimdilik

print(f"📍 Bölge: Antalya | Tarih: {start_date} - {end_date}")

# 3. Veri Çekme Fonksiyonu
def get_features(feature):
    # Yangının olduğu tarihi al
    date = ee.Date(feature.get('ACQ_DATE'))
    
    # a) Sıcaklık (LST) - MODIS uydusu
    lst = ee.ImageCollection('MODIS/006/MOD11A2') \
        .filterDate(date.advance(-10, 'day'), date.advance(2, 'day')) \
        .mean().select(['LST_Day_1km'], ['LST'])
        
    # b) Bitki Örtüsü (NDVI) - MODIS uydusu
    ndvi = ee.ImageCollection('MODIS/006/MOD13A1') \
        .filterDate(date.advance(-16, 'day'), date.advance(2, 'day')) \
        .mean().select(['NDVI'])
        
    # c) Yükseklik - SRTM uydusu
    srtm = ee.Image('USGS/SRTMGL1_003').select(['elevation'])
    
    # Hepsini tek bir görüntüde birleştir
    full_img = lst.addBands(ndvi).addBands(srtm)
    
    # O noktadaki (koordinattaki) değerleri oku
    stats = full_img.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=feature.geometry(),
        scale=1000
    )
    return feature.set(stats)

# 4. Veri Setini Oluşturma
print("🔥 Yangın verileri NASA FIRMS sunucularından çekiliyor...")

# Yangın Olan Noktalar (Label = 1)
fire_points = ee.ImageCollection('FIRMS') \
    .filterDate(start_date, end_date) \
    .filterBounds(roi)

# Görüntüleri noktalara çevir
dataset = fire_points.map(lambda img: img.reduceToVectors(
    geometry=roi, scale=1000, geometryType='centroid'
)).flatten()

# Etiketle: 1 = Yangın
dataset = dataset.map(lambda f: f.set('label', 1))

# 5. Uydu Verilerini Eşle
print("🛰️ Uydu görüntüleri (Sıcaklık, NDVI, Yükseklik) işleniyor...")
dataset_processed = dataset.map(get_features)

# Boş verileri (bulutlu günler vs) temizle
dataset_final = dataset_processed.filter(ee.Filter.notNull(['LST', 'NDVI', 'elevation']))

# 6. Drive'a Gönder (Export)
print("🚀 Google Drive'a aktarma görevi başlatılıyor...")

task = ee.batch.Export.table.toDrive(
    collection=dataset_final,
    description='Antalya_Yangin_Verisi_Demo',
    fileFormat='CSV',
    selectors=['label', 'LST', 'NDVI', 'elevation', 'ACQ_DATE']
)

task.start()

print("\n✅ GÖREV BAŞARIYLA GÖNDERİLDİ!")
print("------------------------------------------------")
print("Lütfen şu adrese gidip işlemin bitmesini bekle:")
print("👉 https://code.earthengine.google.com/tasks")
print("------------------------------------------------")
print("İşlem bitince (Mavi Tik), Google Drive ana sayfana")
print("'Antalya_Yangin_Verisi_Demo.csv' dosyası gelecek.")