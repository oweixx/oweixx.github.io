# Personal homepage & blog

개인 홈페이지와 Markdown 블로그입니다. 공개 블로그에는 기존 글 중 `Research_rebuttal`, `Research_decision` 두 편만 옮겼습니다.

## 배포 전에 작성하고 확인하기

Windows에서는 **`start-preview.cmd`를 더블클릭**한 후 아래 주소를 엽니다. 서버가 실행 중인 창은 열어두세요. 처음에는 패키지 설치를 위해 인터넷 연결이 필요합니다.

- 홈페이지: <http://127.0.0.1:4321/>
- 블로그: <http://127.0.0.1:4321/blog/>
- 글쓰기 / 실시간 미리보기: <http://127.0.0.1:4321/write/>

일반 개발 환경에서는 Node.js 24 LTS와 npm을 설치하고 다음 명령을 실행합니다.

```sh
npm ci
npm run dev
```

글쓰기 화면에서 제목, 글 주소, 날짜, 본문을 입력하면 오른쪽에 바로 렌더링됩니다. 수식, 코드 강조, 표, 이미지도 확인할 수 있습니다. 입력 내용은 현재 브라우저에 임시 저장되어 새로고침 후 복원됩니다. **장기 보관하려면 Markdown 파일을 다운로드하세요.** 다운로드한 파일은 `src/content/blog/`에 저장합니다. `파일 열기`로 기존 `.md` 파일을 다시 편집할 수 있으며, 수정본은 다운로드해서 원래 파일을 교체합니다.

파일을 저장하면 로컬 서버가 변경을 감지합니다. `/blog/`에서 글을 선택하면 실제 발행 레이아웃으로 확인할 수 있습니다. 외부 편집기에서 Markdown을 직접 수정하는 방식도 지원합니다.

작성 화면과 `글쓰기` 링크는 개발 서버에서만 제공되며, 공개 빌드에는 포함되지 않습니다.

## 새 글 파일 만들기

```sh
npm run new -- my-research-note "새 연구 기록"
```

한국 시간 기준 날짜로 초안 파일을 생성하고 미리보기 주소를 출력합니다. 기존 파일을 덮어쓰지 않습니다.

```yaml
---
title: "새 연구 기록"
date: 2026-10-01
slug: my-research-note
description: "목록에 표시할 짧은 설명"
category: Research
tags: [3DGS, 회고]
draft: true
---

## 시작하며

본문을 작성합니다.
```

- 카테고리: `Papers`, `Research`, `Notes`, `Life`
- 글 주소: `/blog/2026/my-research-note/`. `slug`는 영문, 숫자, `-`, `_`를 사용합니다.
- `draft: true`: 로컬에서만 표시됩니다. 배포할 때 글 페이지와 홈페이지/블로그 목록에서 모두 제외됩니다. `draft`를 생략해도 초안으로 처리합니다.
- `draft: false`: 공개 빌드에 포함됩니다.
- `updated: 2026-10-02`: 선택 사항이며 작성일 이후의 수정일을 표시합니다.
- 이미지는 `assets/blog/`에 저장하고 `![설명](/assets/blog/image.png)`로 삽입합니다.
- 수식은 `$x^2$` 또는 여러 줄의 `$$ ... $$`, 코드는 언어 이름이 있는 Markdown 코드 블록으로 작성합니다.
- 미리보기와 발행 본문이 같은 수식/코드 렌더러를 사용합니다. 본문은 표준 Markdown으로 작성하며 HTML 삽입은 지원하지 않습니다.

초안도 공개 저장소에 커밋하면 GitHub에서 원문을 볼 수 있습니다. 비공개로 보관할 글은 로컬에만 저장하세요.

## 이미지와 갤러리 넣기

로컬 `/write/`에서 **이미지 추가**를 누르면 사진을 본문에 각각 넣습니다. **갤러리 추가**를 누르고 여러 사진을 선택하면 한 영역에서 좌우로 넘겨 보는 갤러리를 넣습니다. 선택한 사진은 `assets/blog/<글 주소>/`에 실제 저장되고 미리보기에서 바로 보입니다. 글 주소를 비워두면 `assets/blog/uploads/`에 저장합니다. 사진별 파일명은 겹치지 않게 생성됩니다.

직접 Markdown을 작성할 수도 있습니다. 일반 이미지에 따옴표로 설명을 넣으면 이미지 아래에 표시됩니다. 외부 HTTPS 이미지 주소도 사용할 수 있습니다.

```markdown
![사진 설명](/assets/blog/my-note/photo.jpg "이미지 아래에 보일 설명")

:::gallery

![첫 번째 사진](/assets/blog/my-note/photo-1.jpg "첫 번째 설명")

![두 번째 사진](/assets/blog/my-note/photo-2.jpg "두 번째 설명")

![세 번째 사진](https://example.com/photo-3.jpg)

:::
```

갤러리는 한 글에 여러 개 넣을 수 있습니다. 갤러리 안에는 이미지 줄과 빈 줄만 넣습니다. 사진 아래에는 따옴표 안의 설명이 표시되고, 따옴표 설명이 없으면 `![…]` 안의 설명을 사용합니다. 좌우 버튼, 이미지 영역에서 좌우 방향키·Home·End, 모바일 스와이프로 이동합니다. 사진을 자르지 않고 표시하며 자동으로 넘어가지는 않습니다.

**배포할 때는 Markdown 파일과 `assets/blog/`의 이미지 파일을 함께 커밋·push하세요.** 글을 삭제해도 이미지 파일은 자동으로 삭제되지 않습니다. 선택 가능한 파일은 PNG/JPEG/GIF/WebP/AVIF, 한 장당 20MB·한 번에 20장/80MB까지입니다. 이미지 추가와 파일 저장 기능은 로컬 글쓰기 화면에서만 제공됩니다.

## 방문 통계

Cloudflare Workers와 SQLite Durable Object에 방문수를 저장합니다. 공개 사이트에서 페이지를 열거나 새로고침하면 한 번 집계하며, 저장된 결과를 응답받는 즉시 화면에 표시합니다. 다른 방문자의 접속도 WebSocket으로 반영됩니다. 로컬 미리보기와 글쓰기 화면은 집계하지 않습니다.

- 홈페이지: 전체 사이트의 `Today`, `Total`.
- 블로그 목록과 글: 해당 글의 누적 `Visits`. 목록을 보는 것만으로 각 글의 Visits가 증가하지 않습니다.
- `/stats/`: 최근 7일·30일 방문 그래프, 홈페이지 누적 방문수, 글별 방문 순위. 모든 방문자가 볼 수 있습니다.

전체 방문수는 페이지를 연 횟수의 합이며 사람 수와는 다릅니다. **한국 시간 자정**에 Today가 초기화됩니다. 통신 재시도는 같은 방문 ID를 사용해 중복 집계를 막으며, 새로고침에는 새 ID를 부여합니다. IP, 쿠키, 개인 식별 정보는 저장하지 않습니다. 실시간 카운터를 연결한 시점부터 집계하며 이전 GoatCounter 기록은 합산하지 않습니다.

연결 주소는 `src/config/analytics.ts`에서 관리합니다. 브라우저에는 공개 주소만 포함되며 Cloudflare 인증 정보는 포함하지 않습니다. 집계 요청이 실패하면 숫자 대신 `—` 또는 안내 문구를 보여줍니다. 공개 카운터는 로그인 없이 집계하므로 고의적인 요청이나 자동화 브라우저 방문도 증가할 수 있습니다.

카운터 서버 코드를 수정할 때는 Node.js 24와 인증된 Cloudflare 계정으로 다음 명령을 실행합니다. 일반 글 수정에는 서버 재배포가 필요하지 않습니다.

```sh
npm ci --prefix services/visits
npm test --prefix services/visits
npm run deploy --prefix services/visits
```

최초 연결 시에는 `npx --prefix services/visits wrangler login`으로 로그인합니다. 설정은 `services/visits/wrangler.jsonc`에 있으며, 이 저장소의 GitHub Actions는 서버 테스트와 사이트 배포를 수행합니다. Workers 서버 배포는 위 명령으로 별도로 실행합니다. 로컬 서버는 `npm run dev --prefix services/visits`로 시작하고 `PUBLIC_VISITS_ENDPOINT=http://127.0.0.1:8787`로 테스트 빌드의 연결 주소를 바꿀 수 있습니다.

## 빌드와 배포

```sh
npm run check
npm test
npm run build
npm run preview
```

`npm run build`는 `dist/`에 정적 페이지와 `assets/`를 생성합니다. `npm run preview`는 이 배포 결과를 로컬에서 확인합니다. 작성 화면과 초안은 이 미리보기에서도 제공되지 않습니다.

배포 준비가 되면 GitHub 저장소의 **Settings → Pages → Source**를 **GitHub Actions**로 설정합니다. 이후 `main`에 push하면 검사와 빌드를 통과한 결과가 GitHub Pages에 배포됩니다. Pull request에서는 검사와 빌드만 실행됩니다. 현재 설정의 배포 주소는 <https://oweixx.github.io/>입니다.

기존 루트 `index.html`의 홈페이지는 `src/pages/index.astro`로 옮겼습니다. 프로필, 뉴스, 논문 정보는 이 파일에서 수정합니다. `assets/`의 기존 파일과 주소는 그대로 사용합니다. `legacy/`는 참고용 보관 디렉터리이며 새 빌드에 포함되지 않습니다.
