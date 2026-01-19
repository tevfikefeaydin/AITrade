# /debug - Hata Ayıklama Ajanı

Bu komut bir hatayla karşılaştığında kullanılır.

## Kullanım
```
/debug [hata mesajı veya dosya]
```

## Adımlar

### 1. 🔍 Hatayı Anla
- Error message'ı parse et
- Stack trace'i analiz et
- Hangi dosya/satır?

### 2. 🎯 Kök Nedeni Bul
- İlgili kodu incele
- Değişken değerlerini kontrol et
- Edge case'leri düşün

### 3. 🧪 Reproduce Et
```python
# Hatayı tekrarlayan minimal kod
def test_reproduce_bug():
    # Setup
    ...
    # Hatayı tetikle
    ...
```

### 4. 💡 Çözüm Öner
- En az 2 farklı çözüm sun
- Her çözümün artı/eksilerini belirt
- Tavsiye edilen çözümü işaretle

### 5. ✅ Düzelt ve Test Et
- Kullanıcı onayladıktan sonra düzelt
- Test yaz (regression önleme)
- Tüm testlerin geçtiğini doğrula

### 6. 📝 Dokümante Et
Eğer bu hata tekrarlanabilir bir pattern ise:
- CLAUDE.md'ye "YAPMA" olarak ekle
- Çözümü "YAP" olarak ekle

## Yaygın Hata Türleri

### ImportError
```bash
pip list | grep [module]
pip install [module]
```

### TypeError / AttributeError
- Type hints kontrol et
- None check ekle

### API Errors
- Rate limit mi?
- Auth sorunu mu?
- Endpoint değişmiş mi?

### Async Errors
- await eksik mi?
- Event loop sorunu mu?

## Çıktı Formatı
```
═══════════════════════════════════════
🐛 DEBUG RAPORU
═══════════════════════════════════════

❌ HATA: [hata tipi]
📍 KONUM: [dosya:satır]
🔍 NEDEN: [açıklama]

💡 ÇÖZÜMLER:
1. [çözüm 1] - ⭐ Tavsiye
2. [çözüm 2]

Hangisini uygulayayım?
═══════════════════════════════════════
```
