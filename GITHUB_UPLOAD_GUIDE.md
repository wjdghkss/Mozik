# GitHub 업로드 가이드

## 1. GitHub 저장소 만들기

GitHub에서 새 저장소를 만듭니다.

추천 저장소 이름:

```text
Mozik
```

또는

```text
mozik-privacy-mosaic
```

README, .gitignore, License는 GitHub에서 따로 추가하지 않는 것을 권장합니다. 이 ZIP에 이미 포함되어 있습니다.

## 2. VS Code에서 폴더 열기

압축을 풀고 `mozik-github-ready` 폴더를 VS Code로 엽니다.

## 3. 원격 저장소 연결 및 업로드

```bash
git init
git remote add origin https://github.com/깃허브아이디/Mozik.git
git add .
git commit -m "Add Mozik privacy mosaic web service"
git branch -M main
git push -u origin main
```

소빈님 계정 예시:

```bash
git remote add origin https://github.com/X0X1N/Mozik.git
```

이미 origin이 있다고 나오면 아래 명령어로 주소를 바꿉니다.

```bash
git remote set-url origin https://github.com/X0X1N/Mozik.git
```

## 4. 업로드 전 개인정보 확인

아래 명령어를 실행해서 실제 DB 비밀번호나 서버 IP가 남아 있지 않은지 확인하세요.

```bash
grep -R "gkrcjf0821" .
grep -R "211.253.27.46" .
grep -R "supersecretkey" .
```

아무 결과도 나오지 않으면 안전합니다.

## 5. GitHub 100MB 제한 주의

GitHub는 100MB가 넘는 단일 파일 업로드를 제한합니다. 현재 포함된 `best.pt`는 약 4MB 수준이라 일반 GitHub 업로드가 가능합니다. 추후 큰 모델 파일을 추가할 경우 Git LFS 사용을 권장합니다.
