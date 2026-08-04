from __future__ import annotations

import json
from urllib.parse import urljoin
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import httpx
from bs4 import BeautifulSoup


DEFAULT_TARGETS = [
    "https://www.k-startup.go.kr/web/contents/bizpbanc-ongoing.do",
    "https://www.kised.or.kr/menu.es?mid=a10205010000",
    "https://www.kosmes.or.kr/nsh/map/main.do#none",
    "https://www.modoo.or.kr",
]

NOISE_TITLE_KEYWORDS = [
    "로그인",
    "회원가입",
    "개인정보",
    "이용약관",
    "저작권",
    "사이트맵",
    "페이스북",
    "인스타그램",
    "유튜브",
    "블로그",
    "TOP",
]

NOISE_URL_KEYWORDS = [
    "facebook.com",
    "instagram.com",
    "youtube.com",
    "blog.naver.com",
    "javascript:",
    "/login",
    "/join",
    "privacy",
    "terms",
]


@dataclass
class WebRecord:
    source_site: str
    url: str
    title: str
    body: str
    links: List[dict]
    notice_id: str = ""
    parent_url: str = ""
    deadline: str = ""
    organization: str = ""
    support_type: str = ""
    region: str = ""


def _site_name(url: str) -> str:
    if "k-startup.go.kr" in url:
        return "k-startup"
    if "kised.or.kr" in url:
        return "kised"
    if "kosmes.or.kr" in url:
        return "kosmes"
    if "modoo.or.kr" in url:
        return "modoo"
    return "unknown"


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_notice_fields(text: str) -> dict:
    deadline = ""
    organization = ""
    support_type = ""
    region = ""

    deadline_patterns = [
        r"마감일자\s*([0-9]{4}-[0-9]{2}-[0-9]{2}|오늘마감)",
        r"마감일자\s*([0-9]{4}\.[ ]?[0-9]{1,2}\.[ ]?[0-9]{1,2}\.)",
        r"신청기간\s*[:：]?\s*([0-9]{4}\.[ ]?[0-9]{1,2}\.[ ]?[0-9]{1,2}\.)\s*[~〜-]\s*([0-9]{4}\.[ ]?[0-9]{1,2}\.[ ]?[0-9]{1,2}\.)",
    ]
    for pattern in deadline_patterns:
        m_deadline = re.search(pattern, text)
        if not m_deadline:
            continue
        if m_deadline.lastindex and m_deadline.lastindex >= 2:
            deadline = f"{m_deadline.group(1)} ~ {m_deadline.group(2)}"
        else:
            deadline = m_deadline.group(1)
        break

    m_org = re.search(r"\b(창업진흥원|중소벤처기업진흥공단|중소벤처기업부|경기테크노파크|울산과학기술원|경기콘텐츠진흥원)\b", text)
    if m_org:
        organization = m_org.group(1)

    m_support = re.search(r"\b(사업화|글로벌|정책자금|창업교육|멘토링ㆍ컨설팅ㆍ교육|인력)\b", text)
    if m_support:
        support_type = m_support.group(1)

    m_region = re.search(r"\b(서울|경기|인천|부산|대구|광주|대전|울산|세종|강원|충북|충남|전북|전남|경북|경남|제주)\b", text)
    if m_region:
        region = m_region.group(1)

    return {
        "deadline": deadline,
        "organization": organization,
        "support_type": support_type,
        "region": region,
    }


def _extract_main_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")

    for bad in soup.select("script, style, noscript, iframe"):
        bad.extract()

    title = _clean_text(soup.title.get_text(" ", strip=True)) if soup.title else "untitled"

    main = soup.select_one("main") or soup.select_one("#contents") or soup.select_one(".contents")
    if main:
        body = _clean_text(main.get_text(" ", strip=True))
    else:
        body = _clean_text(soup.get_text(" ", strip=True))

    return title, body


def _extract_relevant_links(url: str, html: str, max_links: int = 80) -> List[dict]:
    soup = BeautifulSoup(html, "lxml")
    links: List[dict] = []

    for a_tag in soup.select("a"):
        href = (a_tag.get("href") or "").strip()
        text = _clean_text(a_tag.get_text(" ", strip=True))
        if not href or not text:
            continue

        if href.startswith("#") or href.lower().startswith("javascript:"):
            # K-Startup 목록의 go_view(숫자) 패턴은 공고 식별자로 보존한다.
            go_view_match = re.search(r"go_view\((\d+)\)", href)
            if go_view_match:
                links.append({"title": text, "url": "", "notice_id": go_view_match.group(1)})
            continue

        absolute = urljoin(url, href)
        if absolute.startswith("http"):
            links.append({"title": text, "url": absolute, "notice_id": ""})

    dedup = {}
    for link in links:
        key = (link.get("title", ""), link.get("url", ""), link.get("notice_id", ""))
        dedup[key] = link

    deduped = list(dedup.values())
    notice_links = [link for link in deduped if link.get("notice_id")]
    normal_links = [link for link in deduped if not link.get("notice_id")]
    return (notice_links + normal_links)[:max_links]


def _is_noise_link(link: dict) -> bool:
    title = str(link.get("title", "")).lower()
    url = str(link.get("url", "")).lower()

    if len(title) <= 1:
        return True
    if any(keyword.lower() in title for keyword in NOISE_TITLE_KEYWORDS):
        return True
    if any(keyword in url for keyword in NOISE_URL_KEYWORDS):
        return True
    return False


def collect_web_records(urls: List[str] | None = None, timeout_sec: int = 20) -> List[WebRecord]:
    target_urls = urls or DEFAULT_TARGETS
    records: List[WebRecord] = []

    with httpx.Client(follow_redirects=True, timeout=timeout_sec) as client:
        for url in target_urls:
            response = client.get(url)
            response.raise_for_status()
            title, body = _extract_main_text(response.text)
            links = [
                link for link in _extract_relevant_links(url, response.text) if not _is_noise_link(link)
            ]
            if not body:
                continue
            records.append(
                WebRecord(
                    source_site=_site_name(url),
                    url=url,
                    title=title,
                    body=body,
                    links=links,
                )
            )

            # K-Startup 목록 페이지는 notice_id 기반 상세 페이지를 추가 수집한다.
            if _site_name(url) == "k-startup":
                records.extend(_collect_kstartup_details(client=client, list_url=url, links=links))

    return records


def _collect_kstartup_details(client: httpx.Client, list_url: str, links: List[dict], max_details: int = 20) -> List[WebRecord]:
    details: List[WebRecord] = []
    seen_notice_ids = set()

    for link in links:
        notice_id = str(link.get("notice_id", "")).strip()
        link_title = str(link.get("title", "")).strip()
        if not notice_id or not notice_id.isdigit() or notice_id in seen_notice_ids:
            continue
        seen_notice_ids.add(notice_id)

        detail_url = f"{list_url.split('?')[0]}?schM=view&pbancSn={notice_id}"
        try:
            response = client.get(detail_url)
            response.raise_for_status()
        except httpx.HTTPError:
            continue

        title, body = _extract_main_text(response.text)
        if not body:
            continue

        fields = _extract_notice_fields(f"{link_title} {body}")
        details.append(
            WebRecord(
                source_site="k-startup-detail",
                url=detail_url,
                title=title,
                body=body,
                links=[],
                notice_id=notice_id,
                parent_url=list_url,
                deadline=fields["deadline"],
                organization=fields["organization"],
                support_type=fields["support_type"],
                region=fields["region"],
            )
        )

        if len(details) >= max_details:
            break

    return details


def save_records_as_json(records: List[WebRecord], output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized = [
        {
            "source_site": r.source_site,
            "url": r.url,
            "title": r.title,
            "body": r.body,
            "links": r.links,
            "notice_id": r.notice_id,
            "parent_url": r.parent_url,
            "deadline": r.deadline,
            "organization": r.organization,
            "support_type": r.support_type,
            "region": r.region,
        }
        for r in records
    ]
    path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def collect_and_save_default(output_path: str) -> tuple[int, str]:
    records = collect_web_records()
    saved_path = save_records_as_json(records, output_path)
    return len(records), saved_path
