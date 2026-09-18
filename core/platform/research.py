"""Bounded public research retrieval, without shared caches or arbitrary URLs."""
from hashlib import sha256
from xml.etree import ElementTree
import httpx
from core.platform.models import Evidence


async def collect_public_research(query: str) -> list[Evidence]:
    # Only the explicit query leaves the service. Never send source or history.
    async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
        async with client.stream("GET", "https://export.arxiv.org/api/query", params={
            "search_query": query, "start": 0, "max_results": 5,
        }) as response:
            response.raise_for_status()
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) > 1_000_000:
                    raise ValueError("Research response exceeds size limit")
    if b"\x00" in data or b"<!DOCTYPE" in data or b"<!ENTITY" in data:
        raise ValueError("Unexpected XML declarations")
    tree = ElementTree.fromstring(data)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    evidence = []
    for entry in tree.findall("atom:entry", ns)[:5]:
        url = entry.findtext("atom:id", default="", namespaces=ns)
        title = entry.findtext("atom:title", default="", namespaces=ns).strip()
        summary = entry.findtext("atom:summary", default="", namespaces=ns).strip()
        if url and title and summary:
            evidence.append(Evidence(id="arxiv:" + sha256(url.encode()).hexdigest()[:24],
                title=title[:1000], text=summary[:24000], url=url[:2000]))
    return evidence
