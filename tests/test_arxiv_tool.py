"""Tests for tools/arxiv_tool.py."""
from __future__ import annotations

import urllib.error
from unittest.mock import MagicMock, patch

from core.tools.arxiv_tool import (
    _html_to_text,
    _safe_id,
    download_paper_text,
    load_cached_digest,
    save_digest,
    search_arxiv,
)


# ---------------------------------------------------------------------------
# _safe_id
# ---------------------------------------------------------------------------

class TestSafeId:
    def test_alphanumeric_unchanged(self):
        assert _safe_id("abc123") == "abc123"

    def test_dots_replaced(self):
        assert _safe_id("2301.12345") == "2301_12345"

    def test_slashes_replaced(self):
        assert _safe_id("cs/0001234") == "cs_0001234"

    def test_dashes_preserved(self):
        assert _safe_id("2301-12345") == "2301-12345"

    def test_spaces_replaced(self):
        assert _safe_id("abc def") == "abc_def"


# ---------------------------------------------------------------------------
# _html_to_text
# ---------------------------------------------------------------------------

class TestHtmlToText:
    def test_strips_tags(self):
        result = _html_to_text("<p>Hello <b>world</b></p>")
        assert "Hello" in result
        assert "world" in result
        assert "<" not in result

    def test_strips_script_content(self):
        result = _html_to_text("<p>Text</p><script>alert(1)</script><p>More</p>")
        assert "alert" not in result
        assert "Text" in result
        assert "More" in result

    def test_strips_style_content(self):
        result = _html_to_text("<style>body{color:red}</style><p>Content</p>")
        assert "color" not in result
        assert "Content" in result

    def test_collapses_multiple_newlines(self):
        html = "<p>A</p>\n\n\n\n\n<p>B</p>"
        result = _html_to_text(html)
        assert "\n\n\n" not in result

    def test_empty_html(self):
        result = _html_to_text("")
        assert result == ""

    def test_plain_text_preserved(self):
        result = _html_to_text("plain text")
        assert "plain text" in result


# ---------------------------------------------------------------------------
# load_cached_digest / save_digest
# ---------------------------------------------------------------------------

class TestDigestCache:
    def test_load_returns_none_when_missing(self, tmp_path):
        with patch("core.tools.arxiv_tool._PAPERS_CACHE_DIR", tmp_path):
            result = load_cached_digest("2301.99999")
        assert result is None

    def test_save_and_load_roundtrip(self, tmp_path):
        record = {"arxiv_id": "2301.12345", "title": "Test", "digest": "content"}

        with patch("core.tools.arxiv_tool._PAPERS_CACHE_DIR", tmp_path):
            save_digest("2301.12345", record)
            loaded = load_cached_digest("2301.12345")

        assert loaded == record

    def test_save_creates_parent_dir(self, tmp_path):
        nested = tmp_path / "subdir"
        record = {"data": "value"}

        with patch("core.tools.arxiv_tool._PAPERS_CACHE_DIR", nested):
            save_digest("test_id", record)
            assert (nested / "test_id.digest").exists()

    def test_load_corrupt_file_returns_none(self, tmp_path, capsys):
        bad_file = tmp_path / "corrupt.digest"
        bad_file.write_text("not json at all {{", encoding="utf-8")

        with patch("core.tools.arxiv_tool._PAPERS_CACHE_DIR", tmp_path):
            result = load_cached_digest("corrupt")

        assert result is None

    def test_save_uses_safe_id_for_filename(self, tmp_path):
        record = {"title": "test"}
        arxiv_id = "2301.12345"

        with patch("core.tools.arxiv_tool._PAPERS_CACHE_DIR", tmp_path):
            save_digest(arxiv_id, record)
            assert (tmp_path / "2301_12345.digest").exists()


# ---------------------------------------------------------------------------
# search_arxiv
# ---------------------------------------------------------------------------

class TestSearchArxiv:
    def _make_arxiv_result(self, title, abstract, entry_id, short_id, published_date):
        r = MagicMock()
        r.title = title
        r.summary = abstract
        r.entry_id = entry_id
        r.get_short_id.return_value = short_id
        published = MagicMock()
        published.date.return_value.isoformat.return_value = published_date
        r.published = published
        return r

    def test_returns_list_of_dicts(self):
        mock_result = self._make_arxiv_result(
            "Test Paper", "Abstract text.", "http://arxiv.org/abs/2301.12345",
            "2301.12345", "2023-01-01"
        )

        with patch("arxiv.Search") as mock_search:
            mock_search.return_value.results.return_value = iter([mock_result])
            papers = search_arxiv("test query", 5)

        assert len(papers) == 1
        assert papers[0]["title"] == "Test Paper"
        assert papers[0]["arxiv_id"] == "2301.12345"
        assert papers[0]["published"] == "2023-01-01"

    def test_abstract_newlines_replaced(self):
        mock_result = self._make_arxiv_result(
            "T", "line1\nline2", "url", "id", "2023-01-01"
        )

        with patch("arxiv.Search") as mock_search:
            mock_search.return_value.results.return_value = iter([mock_result])
            papers = search_arxiv("q", 1)

        assert "\n" not in papers[0]["abstract"]

    def test_empty_results(self):
        with patch("arxiv.Search") as mock_search:
            mock_search.return_value.results.return_value = iter([])
            papers = search_arxiv("empty query", 5)

        assert papers == []

    def test_max_results_passed_to_arxiv(self):
        with patch("arxiv.Search") as mock_search:
            mock_search.return_value.results.return_value = iter([])
            search_arxiv("q", 17)

        call_kwargs = mock_search.call_args
        assert call_kwargs.kwargs.get("max_results") == 17 or call_kwargs.args[1] == 17


# ---------------------------------------------------------------------------
# download_paper_text
# ---------------------------------------------------------------------------

class TestDownloadPaperText:
    def test_successful_download(self):
        html = b"<html><body><p>Paper content</p></body></html>"
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = html

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = download_paper_text("2301.12345")

        assert result is not None
        assert "Paper content" in result

    def test_http_404_returns_none(self):
        with patch("urllib.request.urlopen",
                   side_effect=urllib.error.HTTPError("url", 404, "Not Found", {}, None)):
            result = download_paper_text("2301.00000")

        assert result is None

    def test_generic_exception_returns_none(self):
        with patch("urllib.request.urlopen", side_effect=Exception("network error")):
            result = download_paper_text("2301.00001")

        assert result is None

