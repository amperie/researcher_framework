import asyncio
import httpx
import pytest
from core.platform.research import collect_public_research


def collect(monkeypatch, xml, status=200):
    constructor = httpx.AsyncClient
    def handle(request):
        assert request.url.host == "export.arxiv.org"
        assert request.url.params["search_query"] == "public query"
        return httpx.Response(status, content=xml)
    monkeypatch.setattr("core.platform.research.httpx.AsyncClient", lambda **kwargs:
        constructor(transport=httpx.MockTransport(handle), **kwargs))
    return asyncio.run(collect_public_research("public query"))


def test_retrieval_returns_inspectable_public_evidence(monkeypatch):
    result = collect(monkeypatch, b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>https://arxiv.org/abs/1</id><title>Paper</title><summary>Abstract</summary></entry></feed>')
    assert result[0].id.startswith("arxiv:")
    assert result[0].text == "Abstract"


@pytest.mark.parametrize("xml", [b"<!DOCTYPE foo><feed/>", b"\x00<feed/>", b"x" * 1000001], ids=["doctype", "wide-encoding", "oversize"])
def test_retrieval_rejects_unsafe_or_oversize_xml(monkeypatch, xml):
    with pytest.raises(ValueError):
        collect(monkeypatch, xml)


def test_research_outage_is_not_reported_as_success(monkeypatch):
    with pytest.raises(httpx.HTTPStatusError):
        collect(monkeypatch, b"unavailable", status=503)
