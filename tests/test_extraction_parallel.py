"""Extraction runs in parallel, and the two things that must not change, do not.

`extract_many` is the first threaded work in the runtime, which is what ADR 0006 bought
the free-threaded interpreter for. Two properties are load-bearing and neither is
guaranteed by a thread pool:

- Results come back in input order, because a plan is portable data and completion order
  is not reproducible.
- The lxml-backed backends still run one at a time on the calling thread. ADR 0006
  accepted lxml under a pinned-off GIL on the grounds that there were no threads; these
  tests are what keep that true now that there are.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from system.components.content_extract import doc_extract, extraction
from system.components.content_extract.extraction import (
    Backend,
    Extraction,
    ExtractOptions,
    Item,
    chain_for,
    extract_many,
    reaches_lxml,
    worker_count,
)

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


class Recorder:
    """A stand-in backend that records which thread ran it, and in what order."""

    def __init__(self) -> None:
        self.threads: dict[str, int] = {}
        self.order: list[str] = []
        self.lock = threading.Lock()

    def __call__(self, item_id: str, path: Path, options: ExtractOptions) -> Extraction:
        with self.lock:
            self.threads[item_id] = threading.get_ident()
            self.order.append(item_id)
        return Extraction(
            item_id=item_id, backend="recorder", text=item_id, char_count=len(item_id)
        )


@pytest.fixture
def recorded(monkeypatch):
    """Replace every backend with a recorder, so no real parser is involved."""

    def install() -> Recorder:
        recorder = Recorder()
        monkeypatch.setattr(
            extraction, "_BACKENDS", {backend: recorder for backend in extraction._BACKENDS}
        )
        return recorder

    return install


def _items(count: int, media_type: str = "application/pdf") -> tuple[Item, ...]:
    return tuple(
        Item(item_id=f"item-{index:03d}", path=Path(f"/nowhere/{index}"), media_type=media_type)
        for index in range(count)
    )


class TestOrder:
    def test_results_come_back_in_input_order(self, recorded) -> None:
        """Not completion order: two runs over one tree must agree entry for entry."""
        recorded()
        items = _items(40)
        results = extract_many(items, max_workers=8)
        assert [result.item_id for result in results] == [item.item_id for item in items]

    def test_order_holds_when_completion_order_is_reversed(self, monkeypatch) -> None:
        finished: list[str] = []

        def backwards(item_id: str, path: Path, options: ExtractOptions) -> Extraction:
            # Later items finish first, so input order and completion order disagree.
            threading.Event().wait(0.02 * (9 - int(item_id.rsplit("-", 1)[1])))
            finished.append(item_id)
            return Extraction(item_id=item_id, backend="backwards", text="x", char_count=1)

        monkeypatch.setattr(
            extraction, "_BACKENDS", {backend: backwards for backend in extraction._BACKENDS}
        )
        items = _items(10)
        results = extract_many(items, max_workers=10)
        assert [result.item_id for result in results] == [item.item_id for item in items]
        assert finished != [item.item_id for item in items], "the premise: completion differed"

    def test_a_serial_run_and_a_parallel_run_agree(self, recorded) -> None:
        recorded()
        items = _items(25)
        serial = extract_many(items, max_workers=1)
        recorded()
        parallel = extract_many(items, max_workers=8)
        assert [result.model_dump() for result in serial] == [
            result.model_dump() for result in parallel
        ]


class TestParallelism:
    def test_threadable_items_really_run_concurrently(self, monkeypatch) -> None:
        """No backend returns until all three pooled items have arrived.

        A barrier is the one assertion here a thread pool cannot fake: if the items ran
        one after another, the first would wait for two arrivals that cannot happen, the
        barrier would break on its timeout, and the test fails rather than passing slowly.
        The first item sits out — it is the warm-up pass, and it runs before the pool.
        """
        arrived = threading.Barrier(3, timeout=10)
        threads: dict[str, int] = {}

        def rendezvous(item_id: str, path: Path, options: ExtractOptions) -> Extraction:
            if item_id != "item-000":
                arrived.wait()
            threads[item_id] = threading.get_ident()
            return Extraction(item_id=item_id, backend="rendezvous", text="x", char_count=1)

        monkeypatch.setattr(
            extraction, "_BACKENDS", {backend: rendezvous for backend in extraction._BACKENDS}
        )
        results = extract_many(_items(4), max_workers=3)
        assert len(results) == 4
        pooled = {item_id: ident for item_id, ident in threads.items() if item_id != "item-000"}
        assert len(set(pooled.values())) == 3, "each pooled item ran on its own thread"

    def test_the_first_item_warms_the_parsers_on_the_calling_thread(self, recorded) -> None:
        """pdfminer interns names into shared tables with a check-then-insert, so the
        standard ones get interned once, here, before anything races for them."""
        recorder = recorded()
        extract_many(_items(8), max_workers=4)
        assert recorder.order[0] == "item-000"
        assert recorder.threads["item-000"] == threading.get_ident()
        assert threading.get_ident() not in {
            ident for item_id, ident in recorder.threads.items() if item_id != "item-000"
        }

    def test_one_worker_stays_on_the_calling_thread(self, recorded) -> None:
        recorder = recorded()
        extract_many(_items(5), max_workers=1)
        assert set(recorder.threads.values()) == {threading.get_ident()}

    def test_a_single_item_never_starts_a_pool(self, recorded) -> None:
        recorder = recorded()
        extract_many(_items(1), max_workers=8)
        assert set(recorder.threads.values()) == {threading.get_ident()}

    def test_worker_count_is_at_least_one_and_never_exceeds_the_measured_knee(self) -> None:
        """One per CPU was measured *slower* than four; see `_WORKER_CEILING`."""
        assert 1 <= worker_count() <= extraction._WORKER_CEILING


class TestLxmlStaysSerial:
    """ADR 0006 accepted undeclared lxml because nothing was threaded. Still true."""

    @pytest.mark.parametrize(
        "media_type", [DOCX, XLSX, "message/rfc822", "application/vnd.ms-outlook"]
    )
    def test_lxml_media_types_are_recognised(self, media_type: str) -> None:
        assert reaches_lxml(Backend.AUTO_CHAIN, media_type)

    @pytest.mark.parametrize("media_type", ["application/pdf", "image/png", "text/plain"])
    def test_other_media_types_are_threadable(self, media_type: str) -> None:
        assert not reaches_lxml(Backend.AUTO_CHAIN, media_type)

    def test_a_named_lxml_backend_is_recognised_without_a_chain(self) -> None:
        assert reaches_lxml(Backend.DOCX, "application/octet-stream")
        assert not reaches_lxml(Backend.PDF_TEXT, "application/octet-stream")

    def test_lxml_items_run_on_the_calling_thread(self, recorded) -> None:
        recorder = recorded()
        items = (
            Item("doc", Path("/nowhere/a"), DOCX),
            Item("pdf", Path("/nowhere/b"), "application/pdf"),
            Item("sheet", Path("/nowhere/c"), XLSX),
        )
        extract_many(items, max_workers=8)
        assert recorder.threads["doc"] == threading.get_ident()
        assert recorder.threads["sheet"] == threading.get_ident()

    def test_lxml_items_finish_before_any_worker_starts(self, recorded) -> None:
        recorder = recorded()
        items = (*_items(6), Item("doc", Path("/nowhere/d"), DOCX))
        extract_many(items, max_workers=6)
        assert recorder.order[0] == "doc", "lxml runs first, alone"

    def test_an_all_lxml_corpus_never_starts_a_pool(self, recorded) -> None:
        recorder = recorded()
        extract_many(_items(6, DOCX), max_workers=8)
        assert set(recorder.threads.values()) == {threading.get_ident()}


class TestAccounting:
    def test_a_worker_that_raises_still_yields_an_item(self, monkeypatch) -> None:
        """Every input ends in exactly one terminal state, threads or no threads."""

        def explode(item_id: str, path: Path, options: ExtractOptions) -> Extraction:
            raise MemoryError("not the kind a backend catches")

        monkeypatch.setattr(
            extraction, "_BACKENDS", {backend: explode for backend in extraction._BACKENDS}
        )
        results = extract_many(_items(8), max_workers=4)
        assert len(results) == 8
        assert all(result.unavailable_reason == "unreadable:MemoryError" for result in results)

    def test_an_unsupported_media_type_is_still_returned(self, recorded) -> None:
        recorded()
        results = extract_many(
            (Item("odd", Path("/nowhere/x"), "application/x-nonsense"),), max_workers=4
        )
        assert results[0].unavailable_reason == "unsupported_media_type:application/x-nonsense"

    def test_an_empty_corpus_is_empty(self, recorded) -> None:
        recorded()
        assert extract_many((), max_workers=4) == ()

    def test_chain_for_a_named_backend_is_just_that_backend(self) -> None:
        assert chain_for(Backend.PDF_OCR, "application/pdf") == (Backend.PDF_OCR,)


class TestTheOperator:
    def test_the_operator_counts_what_it_returns(self, recorded) -> None:
        recorded()
        result = doc_extract.run(
            {
                "root": "/nowhere",
                "items": [
                    {"item_id": "a", "relpath": "a.pdf", "media_type": "application/pdf"},
                    {"item_id": "b", "relpath": "b.pdf", "media_type": "application/pdf"},
                ],
            },
            Path("/nowhere"),
        )
        assert [item["item_id"] for item in result["extractions"]] == ["a", "b"]
        assert result["usable"] == 2
        assert result["unavailable"] == 0

    def test_max_workers_is_not_a_configuration_slot(self) -> None:
        """A subagent selects a group from the closed catalog; it cannot ask for threads."""
        assert "max_workers" not in doc_extract.FEATURES
