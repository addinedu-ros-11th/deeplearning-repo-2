# 팀 Git/GitHub 협업 가이드 (main + develop + feature)

이 문서는 팀 프로젝트를 **원본 저장소(Organization Repository)** 에서 브랜치/PR 기반으로 운영하기 위한 최소 규칙을 정리합니다.

---

## 1) 브랜치 전략

- `main`: 배포/제출/최종 결과물 브랜치 (보호 브랜치)
- `develop` (또는 `dev`): 통합 개발 브랜치 (보호 브랜치)
- `feature/*`: 기능 개발 브랜치 (개인/작업 단위)
- `fix/*`: 버그 수정 브랜치
- `docs/*`: 문서 수정 브랜치
- `hotfix/*`: `main` 긴급 수정 브랜치 (필요할 때만)

권장: **평소 작업은 `develop` 기준으로 `feature/*` 생성 → PR로 `develop`에 병합 → 릴리즈 시 `develop` → `main` PR**.

---

## 2) 기본 규칙 (필수)

- `main`, `develop`에는 **직접 push 금지** (Branch protection 권장)
- 모든 변경은 **PR(Pull Request)로만 병합**
- PR은 최소 **1명 이상 리뷰 승인** 후 병합
- PR은 **작게** 쪼개기 (가능하면 200~400줄 내외 권장)
- 브랜치명/커밋/PR 설명은 **의도를 드러나게** 작성

---

## 3) 브랜치 네이밍 규칙

형식 예시(택1로 통일 권장):

- `feature/<짧은-설명>` 예: `feature/cctv-inference-pipeline`
- `feature/<이슈번호>-<짧은-설명>` 예: `feature/12-add-yolo-train-script`
- `fix/<짧은-설명>` 예: `fix/nms-bug`
- `docs/<짧은-설명>` 예: `docs/update-readme`

금지:

- `feature/test`, `feature/aaa` 같은 의미 없는 이름
- 한 브랜치에 여러 기능/이슈를 섞는 작업

---

## 4) 커밋 메시지 규칙 (간단 버전)

형식: `<type>: <summary>`

`type` 예시:
- `feat`: 기능 추가
- `fix`: 버그 수정
- `docs`: 문서
- `refactor`: 리팩터링(동작 변경 없음)
- `chore`: 빌드/설정/잡무
- `test`: 테스트

예:
- `feat: add dataset split script`
- `fix: handle empty frame in detector`
- `docs: update setup instructions`

---

## 5) 작업 시작/업데이트 흐름 (명령어 예시)

### 새 작업 시작 (feature 브랜치 만들기)

```bash
git checkout develop
git pull origin develop
git checkout -b feature/<작업명>
```

### 작업 중 `develop` 변경사항 가져오기 (충돌 최소화)

```bash
git fetch origin
git checkout feature/<작업명>
git rebase origin/develop
```

> 팀 규칙이 rebase가 부담이면 `git merge origin/develop`로 통일해도 됩니다. (팀에서 1개 방식만 채택)

---

## 6) PR 규칙 (권장 템플릿)

PR 제목:
- `[feat] ...`, `[fix] ...`, `[docs] ...` 처럼 한눈에 보이게

PR 본문에 포함:
- 변경 요약 (무엇을/왜)
- 관련 이슈/작업 링크
- 테스트 방법/결과 (예: 실행 커맨드, 스크린샷)
- 리뷰어가 봐야 할 포인트/리스크

---

## 7) 병합(merge) 방식

팀 규칙으로 아래 중 **하나만** 고정 권장:

- **Squash merge**: PR 단위로 히스토리 깔끔 (추천)
- **Merge commit**: 브랜치 히스토리 보존

추가 권장:
- PR 병합 후 브랜치는 GitHub에서 삭제
- 로컬에서도 작업 종료 후 정리

```bash
git checkout develop
git pull origin develop
git branch -d feature/<작업명>
git fetch -p
```

---

## 8) 릴리즈/제출 흐름 (develop → main)

릴리즈(또는 제출) 시점:

1. `develop`에서 테스트/검증 완료
2. `develop` → `main` PR 생성
3. 병합 후 태그(선택): `v0.1`, `v1.0` 등

---

## 9) Hotfix (긴급 수정)

`main`에서 심각한 문제가 발생했을 때만 사용:

```bash
git checkout main
git pull origin main
git checkout -b hotfix/<설명>
# 수정 후 PR -> main
```

`main`에 hotfix가 들어가면 같은 변경을 `develop`에도 반영:
- `main` → `develop` PR 또는 `cherry-pick`으로 동기화

---

## 10) 금지/주의 사항

- 비밀키/토큰/개인정보를 커밋하지 않기 (`.env`, `*.pem`, `credentials.*` 등)
- 대용량 모델/데이터는 Git에 직접 올리지 않기 (필요 시 Git LFS 또는 외부 스토리지)
- 자동 생성물(빌드 산출물, 캐시)은 `.gitignore`로 제외
- 충돌(conflict) 해결이 어려우면 혼자 오래 끌지 말고 바로 공유

---

## 11) 합의해야 할 체크리스트

- [ ] `main`, `develop` 보호 브랜치 설정 여부
- [ ] PR 최소 승인 인원(1명/2명)
- [ ] 병합 방식(Squash vs Merge commit)
- [ ] 이슈 트래킹 규칙(이슈 번호를 브랜치/PR에 넣을지)
- [ ] 릴리즈 기준(언제 `develop` → `main` 올릴지)

---

## 12) 변경사항 병합시 명령어

- `git stash push -u -m "wip before pull"`  
  현재 변경사항(추적/미추적 파일 포함)을 임시 보관. `-u`가 `test.html` 같은 미추적 파일까지 포함.

- `git pull --ff-only origin develop`  
  원격 develop을 “빨리감기”로만 받음. 병합 커밋 없이 안전하게 업데이트.

- `git stash pop`  
  임시 보관했던 변경사항을 다시 꺼내 적용. 충돌 나면 여기서 확인.


