import pandas as pd
import os

# --- AYARLAR ---
# Verilerin olduğu klasör
DATA_FOLDER = 'data'

# Okunacak Dosyalar (Data klasörünün içinde olduklarını varsayıyoruz)
FIRE_DATA_PATH = os.path.join(DATA_FOLDER, 'Antalya_Yangin_Verisi_Tam.csv')
NON_FIRE_DATA_PATH = os.path.join(DATA_FOLDER, 'Antalya_NonFire_Verisi_Final.csv')

# Kaydedilecek Dosya (Bu da Data klasörüne gidecek)
OUTPUT_PATH = os.path.join(DATA_FOLDER, 'Antalya_Merged_Dataset.csv')

def load_and_process_data():
    print(f"📂 Çalışma dizini: {os.getcwd()}")
    print(f"📂 '{DATA_FOLDER}' klasöründeki veriler işleniyor...")

    # 1. Dosya Kontrolü
    if not os.path.exists(FIRE_DATA_PATH) or not os.path.exists(NON_FIRE_DATA_PATH):
        print("❌ HATA: CSV dosyaları bulunamadı!")
        print(f"Lütfen şu dosyaların '{DATA_FOLDER}' klasöründe olduğundan emin ol:")
        print(f" - {FIRE_DATA_PATH}")
        print(f" - {NON_FIRE_DATA_PATH}")
        return

    # 2. Verileri Oku
    try:
        df_fire = pd.read_csv(FIRE_DATA_PATH)
        df_non_fire = pd.read_csv(NON_FIRE_DATA_PATH)
        print(f"✅ Yangın Verisi Okundu: {len(df_fire)} satır")
        print(f"✅ Normal Veri Okundu: {len(df_non_fire)} satır")
    except Exception as e:
        print(f"❌ Beklenmedik hata: {e}")
        return

    # 3. Etiketle (1: Yangın, 0: Temiz)
    df_fire['label'] = 1
    df_non_fire['label'] = 0

    # 4. Birleştir
    df_combined = pd.concat([df_fire, df_non_fire], axis=0)

    # 5. Karıştır (Shuffle)
    # frac=1 tüm veriyi alır, random_state=42 her çalıştırmada aynı karıştırmayı yapar
    df_shuffled = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # 6. Temizlik (Boş verileri at)
    print(f"🧹 Temizlik öncesi toplam: {len(df_shuffled)}")
    df_final = df_shuffled.dropna()
    print(f"✨ Temizlik sonrası toplam: {len(df_final)}")

    # 7. Kaydet (Data klasörüne)
    df_final.to_csv(OUTPUT_PATH, index=False)
    
    print("-" * 40)
    print(f"🚀 İŞLEM BAŞARILI!")
    print(f"📂 Dosya şuraya kaydedildi: {OUTPUT_PATH}")
    print("-" * 40)
    print("İlk 5 satır örneği:")
    print(df_final.head())

if __name__ == "__main__":
    load_and_process_data()