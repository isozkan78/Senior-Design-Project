import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Makine Öğrenmesi Kütüphaneleri
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

# Derin Öğrenme (TensorFlow/Keras)
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- AYARLAR ---
DATA_PATH = 'data/Antalya_Merged_Dataset.csv'
MODEL_PATH = 'models/fire_prediction_model.h5'
SCALER_PATH = 'models/scaler.pkl'
RESULTS_FOLDER = 'results'

# Sonuçların kaydedileceği klasörü oluştur
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def evaluate_all_models():
    print("📊 MODELLERİN KARŞILAŞTIRMALI ANALİZİ BAŞLIYOR...")
    print("-" * 50)

    # 1. VERİYİ HAZIRLA
    # ---------------------------------------------------------
    if not os.path.exists(DATA_PATH):
        print(f"❌ HATA: Veri dosyası bulunamadı: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    X = df[['LST', 'NDVI', 'elevation']].values
    y = df['label'].values

    # Scaler'ı yükle (Eğitimde kullanılanın aynısı olmalı)
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)
        print("✅ Kayıtlı Scaler yüklendi ve veri normalize edildi.")
    else:
        print("⚠️ UYARI: Kayıtlı Scaler bulunamadı, yeni scaler oluşturuluyor...")
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

    # Eğitim ve Test setini ayır (train_model.py ile AYNI random_state=42 olmalı)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    print(f"📊 Test Verisi Sayısı: {len(X_test)}")

    # 2. MODELLERİ EĞİT VE TAHMİN AL
    # ---------------------------------------------------------
    
    # --- A) Lojistik Regresyon (Klasik İstatistiksel Yöntem) ---
    print("\n🔹 Model 1: Lojistik Regresyon (LR) eğitiliyor...")
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    y_prob_lr = lr_model.predict_proba(X_test)[:, 1] # Olasılık değerleri (ROC için)

    # --- B) Random Forest (Güçlü Makine Öğrenmesi) ---
    print("🔹 Model 2: Random Forest (RF) eğitiliyor...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

    # --- C) LSTM (Senin Derin Öğrenme Modelin) ---
    print("🔹 Model 3: LSTM Modeli yükleniyor...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ HATA: LSTM modeli bulunamadı: {MODEL_PATH}")
        print("Lütfen önce 'src/train_model.py' kodunu çalıştırın.")
        return

    lstm_model = load_model(MODEL_PATH)
    
    # LSTM 3 boyutlu veri ister: (Örnek, Zaman Adımı, Özellik)
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    y_prob_lstm = lstm_model.predict(X_test_lstm, verbose=0).flatten()
    y_pred_lstm = (y_prob_lstm > 0.5).astype(int) # %50 üzerini 1 kabul et

    # 3. SONUÇLARI HESAPLA VE TABLO OLUŞTUR
    # ---------------------------------------------------------
    models = {
        'Logistic Regression': (y_test, y_pred_lr, y_prob_lr),
        'Random Forest': (y_test, y_pred_rf, y_prob_rf),
        'LSTM (Deep Learning)': (y_test, y_pred_lstm, y_prob_lstm)
    }

    print("\n🏆 PERFORMANS TABLOSU (TÜBİTAK Raporu İçin)")
    print("=" * 75)
    print(f"{'Model Adı':<25} | {'Accuracy':<10} | {'F1-Score':<10} | {'AUC Score':<10}")
    print("-" * 75)

    for name, (y_true, y_pred, y_prob) in models.items():
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        
        print(f"{name:<25} | {acc:.4f}     | {f1:.4f}     | {auc:.4f}")

        # Confusion Matrix Çiz
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'{name} - Confusion Matrix')
        plt.ylabel('Gerçek Durum')
        plt.xlabel('Tahmin Edilen')
        plt.tight_layout()
        plt.savefig(f"{RESULTS_FOLDER}/cm_{name.replace(' ', '_')}.png")
        plt.close()

    # 4. ROC EĞRİSİ ÇİZ (En Önemli Grafik)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 8))
    
    for name, (y_true, y_pred, y_prob) in models.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Rastgele Tahmin (Chance)', linestyle='--')
    plt.xlabel('False Positive Rate (Yanlış Alarm Oranı)', fontsize=12)
    plt.ylabel('True Positive Rate (Gerçek Tespit Oranı)', fontsize=12)
    plt.title('ROC Eğrisi - Model Karşılaştırması', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    save_path = f"{RESULTS_FOLDER}/roc_curve_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("\n✅ ANALİZ TAMAMLANDI!")
    print(f"📁 Grafikler '{RESULTS_FOLDER}' klasörüne kaydedildi.")
    print(f"🖼️ Ana Grafik: {save_path}")

if __name__ == "__main__":
    evaluate_all_models()