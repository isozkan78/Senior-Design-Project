import pandas as pd
import numpy as np

# Dosya İsimleri (Eğer dosyaların farklı yerdeyse yolu düzeltmen gerekebilir)
# Örn: 'data/Antalya_Yangin_Verisi_Tam.csv' gibi
FIRE_DATA_PATH = 'Antalya_Yangin_Verisi_Tam.csv'
NON_FIRE_DATA_PATH = 'Antalya_NonFire_Verisi_Final.csv'

def load_and_process_data():
    print("📂 Veri setleri yükleniyor...")
    
    # 1. Verileri Oku
    try:
        df_fire = pd.read_csv(FIRE_DATA_PATH)
        df_non_fire = pd.read_csv(NON_FIRE_DATA_PATH)
        print(f"✅ Yangın Verisi: {len(df_fire)} satır")
        print(f"✅ Normal Veri: {len(df_non_fire)} satır")
    except FileNotFoundError as e:
        print(f"❌ HATA: Dosya bulunamadı! Lütfen CSV dosyalarını proje klasörüne attığından emin ol.\n{e}")
        return

    # 2. Etiket Kontrolü (Garanti olsun diye)
    df_fire['label'] = 1
    df_non_fire['label'] = 0

    # 3. Birleştirme (Merging)
    # İki tabloyu alt alta ekliyoruz
    df_combined = pd.concat([df_fire, df_non_fire], axis=0)

    # 4. Karıştırma (Shuffling) - ÇOK ÖNEMLİ
    # Verileri karıştırmazsak model önce sadece yangınları ezberler, sonra şaşırır.
    # frac=1 tüm veriyi alır, random_state=42 her seferinde aynı şekilde karıştırır.
    df_shuffled = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # 5. Gereksiz veya Bozuk Veri Temizliği
    # Eksik veri varsa (NaN) o satırları uçur
    print(f"🧹 Temizlik öncesi toplam: {len(df_shuffled)}")
    df_final = df_shuffled.dropna()
    print(f"✨ Temizlik sonrası toplam: {len(df_final)}")

    # 6. Kaydetme
    output_filename = 'Antalya_Merged_Dataset.csv'
    df_final.to_csv(output_filename, index=False)
    
    print("-" * 30)
    print(f"🚀 İŞLEM TAMAM! Dosya oluşturuldu: {output_filename}")
    print("İlk 5 satır örneği:")
    print(df_final.head())

if __name__ == "__main__":
    load_and_process_data()