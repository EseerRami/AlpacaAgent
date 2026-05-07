"""News search — was Tavily ($), now Google News RSS (free, no key).

The function signature is unchanged so the rest of the agent code continues
working without modification. The `tavily` parameter is kept for compatibility
but ignored.

Why Google News RSS:
- No API key, no rate limit, free forever
- Returns recent news on any query (ticker, topic, etc.)
- Structured XML response (title, link, pubDate, source)

If Google ever blocks RSS, swap to Yahoo Finance RSS:
  https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US
"""
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from typing import Literal

from utils.logging import get_logger

log = get_logger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; AlpacaAgent/1.0; +https://github.com/EseerRami/AlpacaAgent)"


def _parse_rss_date(s: str) -> datetime | None:
    """RFC 822: 'Wed, 07 May 2026 16:30:00 GMT' -> aware datetime."""
    if not s:
        return None
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"):
        try:
            d = datetime.strptime(s.strip(), fmt)
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            return d
        except Exception:
            continue
    return None


def _google_news_search(query: str, days: int, max_results: int) -> list[dict]:
    """Pull recent news items from Google News RSS, filter by age, return list."""
    qs = urllib.parse.urlencode({
        "q": query,
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    })
    url = f"https://news.google.com/rss/search?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            xml_bytes = resp.read()
    except Exception as e:
        log.warning("google news fetch failed: %s", e)
        return []

    try:
        root = ET.fromstring(xml_bytes)
    except Exception as e:
        log.warning("google news XML parse failed: %s", e)
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out: list[dict] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = item.findtext("pubDate") or ""
        # source is in <source url="...">SourceName</source>
        src_el = item.find("source")
        source_name = (src_el.text or "").strip() if src_el is not None else ""
        # description has html-escaped snippets — strip tags for `content`
        desc = item.findtext("description") or ""
        # crude tag stripping (keeps text only)
        import re
        clean_desc = re.sub(r"<[^>]+>", " ", desc).strip()
        clean_desc = re.sub(r"\s+", " ", clean_desc)[:500]

        published = _parse_rss_date(pub)
        if published and published < cutoff:
            continue
        out.append({
            "title": title,
            "url": link,
            "content": clean_desc,
            "source": source_name,
            "published_date": published.isoformat() if published else pub,
            "score": None,  # tavily had a score; not applicable
        })
        if len(out) >= max_results:
            break
    return out


def web_search(
    tavily,  # kept for backwards compat; unused
    *,
    query: str,
    topic: Literal["general", "news", "finance"] = "news",
    days: int = 7,
    max_results: int = 5,
) -> dict:
    """Replacement for Tavily.search — same shape, free Google News backend."""
    log.info(
        "GoogleNews web_search: query=%r topic=%s days=%s max_results=%s",
        query, topic, days, max_results,
    )

    # When topic is "finance" or "news", bias the query toward financial sources
    q = query
    if topic == "finance" and "stock" not in q.lower() and "earnings" not in q.lower():
        q = f"{query} stock"

    results = _google_news_search(q, days=days, max_results=max_results)
    urls = [r.get("url") for r in results[:5]]
    log.info("GoogleNews web_search: results=%s urls=%s", len(results), urls)

    res = {"query": query, "results": results, "answer": None}
    log.info("GoogleNews web_search output=%s", json.dumps(res, ensure_ascii=False)[:1500])
    return res
