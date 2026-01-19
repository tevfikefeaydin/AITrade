# /new-feature - Yeni Özellik Ekleme Workflow

Bu komut yeni bir özellik eklemek için kullanılır. Plan-first yaklaşımı kullanır.

## Kullanım
```
/new-feature [özellik açıklaması]
```

## Adımlar

### 1. 🎯 Plan Modu (ZORUNLU)
Önce plan yap, kod yazmadan önce kullanıcıyla anlaş:

- Özelliğin amacı nedir?
- Hangi dosyalar etkilenecek?
- Hangi bağımlılıklar gerekli?
- API endpoint'leri?
- Test stratejisi?

**Plan onaylanmadan KOD YAZMA!**

### 2. 🌿 Branch Oluştur
```bash
git checkout -b feature/[ozellik-adi]
```

### 3. 📁 Dosya Yapısı
Gerekli dosyaları oluştur:
- `src/[module]/[feature].py` - Ana kod
- `tests/test_[feature].py` - Testler
- `docs/[feature].md` - Dokümantasyon (opsiyonel)

### 4. 🧪 TDD Yaklaşımı
1. Önce test yaz (RED)
2. Kodu yaz (GREEN)
3. Refactor et (REFACTOR)

### 5. 📝 CLAUDE.md Güncelle
Yeni özelliği dokümante et:
- Proje yapısına ekle
- Öğrenilen dersleri not al

### 6. ✅ Son Kontroller
- [ ] Testler geçiyor mu?
- [ ] Lint hataları yok mu?
- [ ] Type hints var mı?
- [ ] Docstring var mı?

### 7. 🚀 Commit & Push
```bash
git add -A
git commit -m "feat: [özellik açıklaması]"
git push origin HEAD
```

## Örnek
```
/new-feature RSI divergence tespiti ekle

Plan:
1. src/analysis/indicators/rsi.py'ye divergence fonksiyonu ekle
2. Bullish ve bearish divergence tespiti
3. tests/test_rsi.py'ye testler ekle
4. Signal generator'a entegre et

Onaylıyor musun?
```
