# /pr - Pull Request Oluştur

Bu komut PR oluşturmadan önce tüm kontrolleri yapar.

## Ön Kontroller

### 1. Branch Kontrolü
- main/master'da mısın? → "Önce yeni branch oluştur!"
- Branch adı uygun mu? → `feature/`, `fix/`, `refactor/` prefix

### 2. Kod Kalitesi
```bash
ruff check src/
black --check src/
```

### 3. Testler
```bash
pytest tests/ -v
```

### 4. Commit Geçmişi
- Commitler anlamlı mı?
- Squash gerekiyor mu?

## PR Oluşturma

Tüm kontroller geçtiyse:

```bash
# GitHub CLI ile
gh pr create --title "$PR_TITLE" --body "$PR_BODY"
```

## PR Template

```markdown
## 📝 Değişiklik Özeti
[Bu PR ne yapıyor?]

## 🔄 Değişiklik Tipi
- [ ] Yeni özellik (feat)
- [ ] Bug düzeltme (fix)
- [ ] Refactoring
- [ ] Dokümantasyon
- [ ] Test

## ✅ Kontrol Listesi
- [ ] Testler yazıldı
- [ ] Lint hataları yok
- [ ] Dokümantasyon güncellendi
- [ ] CLAUDE.md güncellendi (gerekiyorsa)

## 📸 Ekran Görüntüleri (varsa)
[UI değişikliklerinde ekle]

## 🔗 İlgili Issue
Closes #[issue_number]
```
