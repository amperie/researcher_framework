"""Tests for tools/arxiv_tool.py.

All tests mock `arxiv.Search` so no network calls are made.
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from tools.arxiv_tool import get_paper_details, search_arxiv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    title: str = "Test Paper",
    summary: str = "Line one.\nLine two.",
    entry_id: str = "http://arxiv.org/abs/2310.01234v1",
    short_id: str = "2310.01234",
    date_str: str = "2023-10-02",
    author_names: list[str] | None = None,
    pdf_url: str = "http://arxiv.org/pdf/2310.01234v1",
    categories: list[str] | None = None,
) -> MagicMock:
    """Return a MagicMock that quacks like an arxiv.Result."""
    r = MagicMock()
    r.title = title
    r.summary = summary
    r.entry_id = entry_id
    r.get_short_id.return_value = short_id
    r.published.date.return_value.isoformat.return_value = date_str
    r.pdf_url = pdf_url
    r.categories = categories or ["cs.LG", "cs.AI"]

    names = author_names or ["Alice Example", "Bob Sample"]
    r.authors = [MagicMock(name=n) for n in names]
    # MagicMock uses 'name' as a special attribute — set it explicitly
    for mock_author, name in zip(r.authors, names):
        mock_author.name = name

    return r


def _patch_search(results: list[MagicMock]):
    """Return a context manager that patches arxiv.Search to return *results*."""
    mock_search_cls = MagicMock()
    mock_search_cls.return_value.results.return_value = iter(results)
    return patch("tools.arxiv_tool.arxiv.Search", mock_search_cls)


# ---------------------------------------------------------------------------
# search_arxiv
# ---------------------------------------------------------------------------

class TestSearchArxiv:
    def test_returns_list_of_dicts(self):
        result = _make_result()
        with _patch_search([result]):
            papers = search_arxiv("mechanistic interpretability", max_results=5)

        assert isinstance(papers, list)
        assert len(papers) == 1

    def test_dict_has_required_keys(self):
        result = _make_result()
        with _patch_search([result]):
            papers = search_arxiv("any query", max_results=1)

        paper = papers[0]
        assert set(paper.keys()) == {"title", "abstract", "url", "arxiv_id", "published"}

    def test_newlines_stripped_from_abstract(self):
        result = _make_result(summary="First line.\nSecond line.\nThird.")
        with _patch_search([result]):
            papers = search_arxiv("query", max_results=1)

        assert "\n" not in papers[0]["abstract"]
        assert "First line. Second line. Third." == papers[0]["abstract"]

    def test_field_values_match_result(self):
        result = _make_result(
            title="My Paper",
            entry_id="http://arxiv.org/abs/2401.99999v2",
            short_id="2401.99999",
            date_str="2024-01-15",
        )
        with _patch_search([result]):
            papers = search_arxiv("query", max_results=1)

        p = papers[0]
        assert p["title"] == "My Paper"
        assert p["url"] == "http://arxiv.org/abs/2401.99999v2"
        assert p["arxiv_id"] == "2401.99999"
        assert p["published"] == "2024-01-15"

    def test_multiple_results_returned_in_order(self):
        results = [
            _make_result(title="Paper A", short_id="2310.00001"),
            _make_result(title="Paper B", short_id="2310.00002"),
            _make_result(title="Paper C", short_id="2310.00003"),
        ]
        with _patch_search(results):
            papers = search_arxiv("query", max_results=3)

        assert len(papers) == 3
        assert [p["title"] for p in papers] == ["Paper A", "Paper B", "Paper C"]

    def test_empty_results(self):
        with _patch_search([]):
            papers = search_arxiv("niche obscure topic xyz", max_results=5)

        assert papers == []

    def test_search_called_with_correct_args(self):
        mock_search_cls = MagicMock()
        mock_search_cls.return_value.results.return_value = iter([])

        with patch("tools.arxiv_tool.arxiv.Search", mock_search_cls) as mock_cls, \
             patch("tools.arxiv_tool.arxiv.SortCriterion") as mock_criterion:
            search_arxiv("activation patching", max_results=7)

        mock_search_cls.assert_called_once_with(
            query="activation patching",
            max_results=7,
            sort_by=mock_criterion.Relevance,
        )

    def test_max_results_respected(self):
        """Verifies max_results is forwarded — the arxiv lib enforces the cap."""
        results = [_make_result(short_id=f"2310.{i:05d}") for i in range(10)]
        mock_search_cls = MagicMock()
        mock_search_cls.return_value.results.return_value = iter(results)

        with patch("tools.arxiv_tool.arxiv.Search", mock_search_cls):
            search_arxiv("query", max_results=10)

        _, kwargs = mock_search_cls.call_args
        assert kwargs["max_results"] == 10


# ---------------------------------------------------------------------------
# get_paper_details
# ---------------------------------------------------------------------------

class TestGetPaperDetails:
    def test_returns_dict_with_all_keys(self):
        result = _make_result()
        with _patch_search([result]):
            details = get_paper_details("2310.01234")

        expected_keys = {"title", "abstract", "url", "arxiv_id", "published",
                         "authors", "pdf_url", "categories"}
        assert set(details.keys()) == expected_keys

    def test_authors_are_strings(self):
        result = _make_result(author_names=["Alice Example", "Bob Sample"])
        with _patch_search([result]):
            details = get_paper_details("2310.01234")

        assert details["authors"] == ["Alice Example", "Bob Sample"]

    def test_pdf_url_and_categories(self):
        result = _make_result(
            pdf_url="http://arxiv.org/pdf/2310.01234v1",
            categories=["cs.LG", "stat.ML"],
        )
        with _patch_search([result]):
            details = get_paper_details("2310.01234")

        assert details["pdf_url"] == "http://arxiv.org/pdf/2310.01234v1"
        assert details["categories"] == ["cs.LG", "stat.ML"]

    def test_newlines_stripped_from_abstract(self):
        result = _make_result(summary="Part one.\nPart two.")
        with _patch_search([result]):
            details = get_paper_details("2310.01234")

        assert "\n" not in details["abstract"]

    def test_search_called_with_id_list(self):
        result = _make_result()
        mock_search_cls = MagicMock()
        mock_search_cls.return_value.results.return_value = iter([result])

        with patch("tools.arxiv_tool.arxiv.Search", mock_search_cls):
            get_paper_details("2401.99999")

        mock_search_cls.assert_called_once_with(id_list=["2401.99999"])

    def test_raises_stop_iteration_when_not_found(self):
        """arxiv.Search returns an empty iterator for unknown IDs — next() raises."""
        with _patch_search([]):
            with pytest.raises(StopIteration):
                get_paper_details("0000.00000")
