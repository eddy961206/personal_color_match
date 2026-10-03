# Vercel 배포

## 배포 설정

저장소 루트의 `vercel.json`이 배포 설정을 관리해.

| 항목 | 설정 |
| --- | --- |
| Framework Preset | Other (`framework: null`) |
| Root Directory | 저장소 루트 (`.`) |
| Install Command | 빈 문자열: 외부 패키지 설치 생략 |
| Build Command | `npm run check` — 단위 테스트 후 정적 빌드 |
| Output Directory | `dist` |
| 애플리케이션 환경 변수 | 없음 |
| 애플리케이션 API 키 | 없음 |

`npm start`는 로컬 개발용 서버야. Vercel에서 상시 서버로 실행하지 않아. Vercel은 `dist`의 정적 파일만 제공해. Git 이력, 테스트, 문서, Python 실행 파일은 출력 폴더에 복사하지 않아.

CSP, 콘텐츠 유형 검사, 리퍼러 차단, 프레임 삽입 차단 헤더를 설정했어. 해시가 붙지 않은 앱 파일은 HTTP 재검증을 사용하고, 서비스 워커 스크립트에는 별도 재검증 헤더를 적용했어. 브라우저의 앱 셸 오프라인 캐시는 기존 동작을 유지해.

## 기존 GitHub 저장소를 연결해서 배포하기

1. [Vercel 새 프로젝트](https://vercel.com/new)에서 사용할 계정을 선택해.
2. `eddy961206/personal_color_match` 옆의 **Import**를 눌러. 저장소가 보이지 않으면 Vercel의 GitHub 앱 설정에서 이 저장소의 접근을 허용해.
3. Root Directory를 저장소 루트로 유지하고 **Deploy**를 눌러. 별도 환경 변수나 유료 분석 API를 추가할 필요는 없어.
4. Vercel의 배포 상태가 **Ready**가 된 다음 제공되는 실제 주소를 열어 확인해. 프로젝트명만으로 접속 주소를 추정하지 마.

배포가 성공하면 Vercel의 Git 연결 설정에 따라 이후 커밋을 자동 배포할 수 있어. 저장소를 복제하는 Deploy Button 대신 기존 저장소를 Import하면 원래 저장소와 연결을 유지할 수 있어.

호스팅의 요금·사용량 정책은 Vercel 계정의 플랜에 따라 달라져. 앱에 API 호출 비용이 없다는 사실과 호스팅 비용은 구분해야 해.

## 2026-10-03 확인 이력

- 앱 기준 커밋: `17c6891941be80fd139b124b0187e5c4a0e4767a`.
- Vercel 설정 커밋: `710c3ae0f3998f2f60a5030d09795ab7baa3ff2f`.
- 첨부 소스 중 실행·빌드·테스트에 사용하는 18개 파일의 Git blob SHA를 현재 GitHub 소스와 대조했어.
- Node.js 22.16.0에서 `npm run check`를 실행했고, 기존 단위 테스트 59개가 모두 통과했어.
- 정적 빌드 성공: revision `93d0de5ecfbc`, 앱 파일 11개, 원본 합계 76,266바이트, 파일별 gzip 합계 27,879바이트. gzip 합계는 실제 Vercel 전송량을 측정한 값이 아니야.
- **Vercel 배포는 완료하지 못했어.** 연결된 계정과 프로젝트 조회는 성공했지만 `deploy_to_vercel` 호출은 JSON-RPC `-32602`, `Tool deploy_to_vercel not found`를 반환했어. 도구 정의를 다시 확인해도 실행 가능한 파일 업로드·프로젝트 생성 인터페이스를 확보하지 못했어.
- CLI에서 사용할 Vercel 인증도 현재 실행 환경에 없어서 임의의 배포 URL을 만들거나 배포 성공을 기록하지 않았어. 계정의 기존 프로젝트는 변경하지 않았어.
- **브라우저 검증도 미완료야.** Chromium이 로컬 주소를 `net::ERR_BLOCKED_BY_ADMINISTRATOR`로 차단했어. 첫 페이지 진입부터 실패했으므로 화면 렌더링, 모바일 조작, 다운로드, 사진 처리, 서비스 워커의 실제 브라우저 동작은 이번 검증에서 통과 처리하지 않았어.

## 실제 배포 후 확인할 항목

- `/`와 앱의 JS·CSS·서비스 워커가 정상 응답하는지 확인해.
- 데스크톱과 모바일에서 색 입력, 팔레트 변경, 저장·복원, 사진의 좌우 색 선택이 작동하는지 확인해.
- 사진 업로드나 외부 분석 요청이 발생하지 않는지 네트워크 패널로 확인해.
- JSON·PNG 다운로드와 오프라인 재접속을 확인해.
- 기존 서비스 워커가 있는 브라우저에서도 새 버전이 정상 적용되는지 확인해.

## 설정 근거

- [Vercel 정적 설정](https://vercel.com/docs/project-configuration/vercel-json)
- [빌드·출력 디렉터리 설정](https://vercel.com/docs/builds/configure-a-build)
- [Git 저장소 연결](https://vercel.com/docs/git)
