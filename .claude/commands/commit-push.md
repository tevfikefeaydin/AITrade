# /commit-push - Git Commit ve Push

Bu komut çalıştırıldığında:

1. Önce git status kontrol et
2. Değişiklikleri analiz et ve anlamlı bir Türkçe commit mesajı oluştur
3. Commit formatı: `[tip]: açıklama`
   - feat: Yeni özellik
   - fix: Bug düzeltme
   - refactor: Kod iyileştirme
   - docs: Dokümantasyon
   - test: Test ekleme
4. Commit at ve push et

```bash
git status
git add -A
git commit -m "$COMMIT_MESSAGE"
git push origin HEAD
```

Eğer push başarısızsa, önce pull --rebase dene.
