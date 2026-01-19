# /simplify - Kod Sadeleştirme Ajanı

Bu komut bir subagent olarak çalışır. Görevi:

## Analiz Et
1. Belirtilen dosya veya modülü incele
2. Kod kalitesi sorunlarını tespit et

## Sadeleştir
- [ ] Kullanılmayan import'ları kaldır
- [ ] Dead code (erişilmeyen kod) temizle
- [ ] Tekrarlanan kodu DRY prensibine göre refactor et
- [ ] Uzun fonksiyonları küçük parçalara böl (max 20 satır)
- [ ] Magic number'ları constant yap
- [ ] Complex conditionals'ı early return ile sadeleştir
- [ ] Nested loops'u optimize et
- [ ] Type hints eksikse ekle
- [ ] Docstring eksikse ekle

## Örnek Sadeleştirmeler

### Önce:
```python
def process(d):
    if d is not None:
        if len(d) > 0:
            result = []
            for i in d:
                if i > 10:
                    result.append(i * 2)
            return result
    return []
```

### Sonra:
```python
def process(data: list[int]) -> list[int]:
    """Verilen listedeki 10'dan büyük değerleri 2 ile çarpar."""
    if not data:
        return []
    return [x * 2 for x in data if x > 10]
```

## Çıktı
- Yapılan değişiklikleri listele
- Her değişikliğin nedenini açıkla
- Değişiklik yapmadan önce kullanıcıya sor
