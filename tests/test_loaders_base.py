# tests/test_loaders_base.py

"""Tests for the loader foundation: model, protocol, registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.loaders.base import (
    BaseLoader,
    LoadedDocument,
    LoadedPage,
    LoaderError,
    LoaderProtocol,
)
from src.loaders.registry import LoaderRegistry


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


class TestLoadedPage:
    def test_minimal_construction(self):
        page = LoadedPage(text="hello")
        assert page.text == "hello"
        assert page.page_number is None
        assert page.metadata == {}

    def test_with_metadata(self):
        page = LoadedPage(text="x", page_number=3, metadata={"sheet_name": "Q1"})
        assert page.page_number == 3
        assert page.metadata["sheet_name"] == "Q1"

    def test_frozen(self):
        page = LoadedPage(text="x")
        with pytest.raises(Exception):
            page.text = "y"  # type: ignore[misc]


class TestLoadedDocument:
    def test_total_text_concatenates(self):
        doc = LoadedDocument(
            source_path="/x.txt",
            document_name="x.txt",
            mime_type="text/plain",
            pages=[LoadedPage("alpha"), LoadedPage("beta"), LoadedPage("gamma")],
        )
        assert doc.total_text == "alpha\n\nbeta\n\ngamma"

    def test_total_chars(self):
        doc = LoadedDocument(
            source_path="/x", document_name="x", mime_type="t/p",
            pages=[LoadedPage("abc"), LoadedPage("de")],
        )
        assert doc.total_chars == 5

    def test_is_empty_when_no_pages(self):
        doc = LoadedDocument(source_path="/x", document_name="x", mime_type="t/p", pages=[])
        assert doc.is_empty is True
        assert doc.total_text == ""

    def test_is_empty_when_all_pages_empty(self):
        doc = LoadedDocument(
            source_path="/x", document_name="x", mime_type="t/p",
            pages=[LoadedPage(""), LoadedPage("")],
        )
        assert doc.is_empty is True

    def test_metadata_default(self):
        doc = LoadedDocument(source_path="/x", document_name="x", mime_type="t/p", pages=[])
        assert doc.metadata == {}


# --------------------------------------------------------------------------- #
# Protocol + BaseLoader
# --------------------------------------------------------------------------- #


class _StubLoader(BaseLoader):
    name = "stub"
    extensions = (".stub",)
    mime_types = ("application/x-stub",)

    def load(self, path: Path) -> LoadedDocument:
        path = self._resolve(path)
        return LoadedDocument(
            source_path=str(path),
            document_name=path.name,
            mime_type=self._mime_for(path),
            pages=[LoadedPage(text="stub content", page_number=None,
                              metadata=self._base_page_metadata(path))],
        )


class TestBaseLoader:
    def test_can_load_by_extension(self, tmp_path):
        p = tmp_path / "f.stub"
        p.touch()
        assert _StubLoader().can_load(p) is True

    def test_can_load_rejects_other_extensions(self, tmp_path):
        p = tmp_path / "f.other"
        p.touch()
        assert _StubLoader().can_load(p) is False

    def test_resolve_raises_for_missing(self):
        with pytest.raises(FileNotFoundError):
            _StubLoader()._resolve(Path("/definitely/does/not/exist/abc.stub"))

    def test_base_page_metadata_has_canonical_keys(self, tmp_path):
        p = tmp_path / "demo.stub"
        p.write_text("x")
        meta = _StubLoader()._base_page_metadata(p, page_number=2)
        assert meta["document_name"] == "demo.stub"
        assert meta["source_path"] == str(p)
        assert meta["page_number"] == 2
        assert meta["loader"] == "_StubLoader"
        assert "mime_type" in meta

    def test_protocol_conformance(self):
        assert isinstance(_StubLoader(), LoaderProtocol)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


class _Loader1(BaseLoader):
    name = "first"
    extensions = (".one",)


class _Loader2(BaseLoader):
    name = "second"
    extensions = (".two",)


class TestRegistry:
    def test_empty(self):
        r = LoaderRegistry()
        assert len(r) == 0

    def test_register_and_find_by_extension(self, tmp_path):
        r = LoaderRegistry()
        r.register(_Loader1())
        p = tmp_path / "a.one"
        p.touch()
        loader = r.find(p)
        assert loader.name == "first"

    def test_first_match_wins(self, tmp_path):
        """When multiple loaders claim a path, registration order decides."""
        class _ClaimEverything(BaseLoader):
            name = "greedy"
            extensions = (".one",)

        r = LoaderRegistry()
        r.register(_Loader1())
        r.register(_ClaimEverything())

        p = tmp_path / "a.one"
        p.touch()
        loader = r.find(p)
        assert loader.name == "first"  # registered first

    def test_no_loader_raises(self, tmp_path):
        r = LoaderRegistry()
        r.register(_Loader1())
        p = tmp_path / "a.unknown"
        p.touch()
        with pytest.raises(LoaderError):
            r.find(p)

    def test_register_rejects_non_baseloader(self):
        r = LoaderRegistry()
        with pytest.raises(TypeError):
            r.register("not a loader")  # type: ignore[arg-type]

    def test_loaders_iteration(self):
        r = LoaderRegistry()
        a, b = _Loader1(), _Loader2()
        r.register(a)
        r.register(b)
        assert list(r.loaders()) == [a, b]
