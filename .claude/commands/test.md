# /test - Test Çalıştır ve Raporla

Bu komut çalıştırıldığında:

1. Virtual environment aktif mi kontrol et
2. pytest ile tüm testleri çalıştır
3. Coverage raporu oluştur
4. Başarısız testleri listele ve nedenlerini açıkla

```bash
# Tüm testler
pytest tests/ -v --tb=short

# Coverage ile
pytest tests/ --cov=src --cov-report=term-missing

# Sadece belirli bir modül
pytest tests/test_$MODULE.py -v
```

Eğer test başarısız olursa:
- Hatanın nedenini analiz et
- Düzeltme önerisi sun
- Kullanıcıya sor: "Düzeltmemi ister misin?"
