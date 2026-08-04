# 테스트 계획 및 결과

## 자동 검사

Pull request와 `main` 브랜치 push에서 다음 검사를 실행합니다.

```bash
ruff check .
pytest -q
```

단위 테스트는 다음 회귀 위험을 우선 검증합니다.

- 명시적 지역 필터가 불일치할 때 빈 결과 반환
- 만 40세 사용자에게 청년 전용 문서 제외
- 양쪽 검색기에 등장한 문서를 RRF에서 우선 배치
- 나이 상한·하한 및 초과·미만 추출
- PDF와 웹 출처 형식 분리
- 중복 출처 제거

## 통합 평가

```bash
python app.py eval
```

통합 평가는 실제 Chroma 인덱스와 OpenAI API 키가 필요합니다. 현재 3개 기본 시나리오를 실행해 다음을 기록합니다.

- `pass_rate`
- `citation_rate`
- `refusal_rate`
- `keyword_hit_ratio`
- `rerank_top_score`
- `answer_preview`

## 해석 시 주의점

키워드 적중은 의미 정확성이나 정책 자격의 정합성을 보장하지 않습니다. 포트폴리오 이후 단계에서는 정답 문서가 표시된 평가 세트를 구축하고 Recall@K, MRR, 인용 정확도와 답변 충실성을 별도로 평가해야 합니다.
