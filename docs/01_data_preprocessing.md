# 데이터 수집 및 전처리

## 입력

- `data/raw/*.pdf`: 정부·지자체 창업지원 공고와 정책자금 문서
- `data/raw/*.json`: 웹 수집기가 만든 레코드 목록

로컬 원문, 인덱스, 전처리 결과와 질의 로그는 Git에서 제외합니다.

## 전처리 파이프라인

1. PDF는 `PyPDFLoader`로 페이지 단위 로딩
2. 웹 레코드는 본문과 유효 링크 문서로 변환
3. `chunk_size=900`, `chunk_overlap=120`으로 분할
4. 각 청크에서 지역·창업 단계·나이 범위 추출
5. 문서 내용과 출처 메타데이터의 SHA-256 해시를 ID로 사용
6. Chroma 컬렉션을 교체하는 전체 재구축 수행

페이지 전체가 아닌 분할된 청크에서 자격 메타데이터를 다시 계산해, 같은 페이지의 다른 문단에 등장한 지역이나 단계가 모든 청크에 잘못 상속되는 현상을 줄입니다.

## 출처 메타데이터

- PDF: `source_kind=pdf`, `source_file`, `page_number`
- 웹: `source_kind=web`, `title`, `source_url`, `notice_id`
- 공통: `region`, `regions`, `stages`, `age_min`, `age_max`, `organization`, `support_type`

웹 문서에는 의미 없는 가상 페이지 번호를 출처로 표시하지 않습니다.

## 향후 개선

- 표를 Markdown/JSON으로 정규화해 별도 인덱싱
- 문서 버전과 수집 시각 보존
- 원문 해시 기반 증분 인덱싱
- OCR이 필요한 스캔 PDF 처리
