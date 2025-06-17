# Turkish Columnist Persona System - Kurulum ve Çalıştırma Kılavuzu

## Gereksinimler

- Python 3.9+
- pip (Python paket yöneticisi)
- Git
- CUDA destekli GPU (önerilen)

## Kurulum Adımları

1. Projeyi klonlayın:
```bash
git clone https://github.com/your-org/turkish-columnist-persona.git
cd turkish-columnist-persona
```

2. Sanal ortam oluşturun ve aktifleştirin:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
.\venv\Scripts\activate  # Windows
```

3. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

4. Model dosyalarını indirin:
```bash
# Model dosyaları models/ klasöründe olmalı
models/
  ├── baris_terkoglu/
  │   └── adapter_model.safetensors
  ├── ahmet_hakan/
  │   ├── adapter_config.json
  │   └── adapter_model.safetensors
  └── abdulkadir_selvi/
      └── adapter_model.safetensors
```

## Çalıştırma Adımları

### 1. Backend Servisini Başlatma

```bash
cd backend
python manage.py runserver
```

Backend servisi varsayılan olarak http://localhost:8000 adresinde çalışacaktır.

### 2. Frontend Uygulamasını Başlatma

```bash
cd frontend
streamlit run app.py
```

Frontend uygulaması varsayılan olarak http://localhost:8501 adresinde çalışacaktır.

### 3. Model Eğitimi ve Değerlendirme

Model eğitimi ve değerlendirme için hazır scriptler kullanılabilir:

```bash
# Tüm modelleri çalıştır
./scripts/run_all.sh

# Sadece RAG modelini çalıştır
./scripts/run_rag.sh

# Sadece LLM modelini çalıştır
./scripts/run_llm.sh

# Sadece fine-tuning işlemini çalıştır
./scripts/run_finetune.sh
```

## Eksik Bileşenler ve Yapılması Gerekenler

1. **Veritabanı Yapılandırması**
   - [ ] PostgreSQL veritabanı kurulumu
   - [ ] Veritabanı şema oluşturma
   - [ ] Migration dosyaları

2. **Model Dosyaları**
   - [ ] Base model dosyalarının indirilmesi
   - [ ] Model konfigürasyon dosyaları
   - [ ] Model ağırlık dosyaları

3. **Çevre Değişkenleri**
   - [ ] `.env` dosyası oluşturma
   - [ ] Gerekli API anahtarları
   - [ ] Veritabanı bağlantı bilgileri

4. **Dokümantasyon**
   - [ ] API dokümantasyonu
   - [ ] Model dokümantasyonu
   - [ ] Deployment kılavuzu

5. **Test Kapsamı**
   - [ ] Unit testler
   - [ ] Integration testler
   - [ ] End-to-end testler

6. **Güvenlik**
   - [ ] SSL/TLS sertifikaları
   - [ ] API rate limiting
   - [ ] Input validation

7. **Monitoring**
   - [ ] Logging sistemi
   - [ ] Performance monitoring
   - [ ] Error tracking

## Hata Ayıklama

### Yaygın Hatalar ve Çözümleri

1. **Model Yükleme Hatası**
   - Model dosyalarının doğru konumda olduğunu kontrol edin
   - CUDA sürücülerinin güncel olduğundan emin olun

2. **Veritabanı Bağlantı Hatası**
   - Veritabanı servisinin çalıştığını kontrol edin
   - Bağlantı bilgilerinin doğru olduğunu kontrol edin

3. **API Bağlantı Hatası**
   - Backend servisinin çalıştığını kontrol edin
   - CORS ayarlarını kontrol edin

## Destek

Teknik destek için:
- GitHub Issues
- Email: support@persona-system.com 