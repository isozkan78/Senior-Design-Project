import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# --- AYARLAR ---
DATA_PATH = 'data/Antalya_Merged_Dataset.csv'
MODEL_SAVE_PATH = 'models/fire_prediction_model.h5'
SCALER_SAVE_PATH = 'models/scaler.pkl'

# Klasörleri oluştur (Yoksa hata verir)
os.makedirs('models', exist_ok=True)

def train_lstm_model():
    print("🚀 Model eğitimi başlıyor...")

    # 1. Veriyi Yükle
    if not os.path.exists(DATA_PATH):
        print(f"❌ HATA: {DATA_PATH} bulunamadı!")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Giriş (X) ve Çıkış (y) olarak ayır
    # ACQ_DATE'i şimdilik eğitime katmıyoruz (İleride zaman serisi olarak ekleyebiliriz)
    X = df[['LST', 'NDVI', 'elevation']].values
    y = df['label'].values

    # 2. Veriyi Ölçekle (0-1 arasına getir)
    # LSTM modelleri büyük sayılarla (örn: 15000) zor çalışır, o yüzden küçültüyoruz.
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Scaler'ı kaydet (Daha sonra tahmin yaparken lazım olacak)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print("✅ Veriler normalize edildi ve Scaler kaydedildi.")

    # 3. LSTM İçin Boyutlandır (Reshape)
    # LSTM [Örnek Sayısı, Zaman Adımı, Özellik Sayısı] formatı ister.
    # Bizim verimiz anlık olduğu için Zaman Adımı = 1 diyoruz.
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # 4. Eğitim ve Test Setine Ayır (%80 Eğitim, %20 Test)
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

    print(f"📊 Eğitim Verisi: {X_train.shape[0]} adet")
    print(f"📊 Test Verisi: {X_test.shape[0]} adet")

    # 5. LSTM Modelini Kur
    model = Sequential()
    
    # Katman 1: LSTM
    model.add(LSTM(64, return_sequences=True, input_shape=(1, 3)))
    model.add(Dropout(0.2)) # Ezberlemeyi önlemek için %20'sini unut
    
    # Katman 2: LSTM
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    
    # Katman 3: Çıktı (Sigmoid: 0 ile 1 arası olasılık verir)
    model.add(Dense(1, activation='sigmoid'))

    # Modeli Derle
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("\n🧠 Model eğitiliyor... (Bu işlem biraz sürebilir)")
    
    # 6. Eğitimi Başlat
    history = model.fit(
        X_train, y_train,
        epochs=50,          # Veriyi kaç kere baştan sona döneceği
        batch_size=32,      # Her seferinde kaç veriyi işleyeceği
        validation_data=(X_test, y_test),
        verbose=1
    )

    # 7. Sonuçları Göster ve Kaydet
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\n🏆 Test Başarısı (Accuracy): %{accuracy * 100:.2f}")

    model.save(MODEL_SAVE_PATH)
    print(f"💾 Model kaydedildi: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_lstm_model()