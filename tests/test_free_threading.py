"""The interpreter this runs on has no GIL, and the stand-ins that made that possible.

Three of our transitive dependencies — orjson, tokenizers, fastuuid — have no
free-threaded build and are imported at module scope by LangGraph and LiteLLM, so
`compat/` supplies pure-Python stand-ins for them. A stand-in that quietly disagrees with
the package it replaces is worse than no stand-in at all: LangGraph serializes every
checkpoint through orjson, so a difference there is a difference in what a resumed run
believes. These tests pin the behaviour that the replaced packages guarantee.

They also assert the interpreter itself, because "the GIL is off" is a claim that decays:
loading an extension that has not declared free-threaded support switches it back on.
"""

from __future__ import annotations

import dataclasses
import datetime
import enum
import json
import math
import subprocess
import sys
import uuid
from pathlib import Path

import fastuuid
import orjson
import pytest
import tokenizers

ROOT = Path(__file__).resolve().parents[1]

#: `vendor/build-tokenizers.sh` produces a real `cp314t` wheel, but only for this platform
#: and this interpreter minor. Everywhere else `compat/tokenizers` stands in, and the two
#: are held to different promises: exact counts, or a refusal to guess at them.
STANDING_IN = hasattr(tokenizers, "TokenizerUnavailable")
needs_the_stand_in = pytest.mark.skipif(not STANDING_IN, reason="the real tokenizers is installed")
needs_the_real_one = pytest.mark.skipif(STANDING_IN, reason="no wheel for this platform")


class TestTheInterpreter:
    def test_the_build_is_free_threaded_and_the_gil_is_off(self) -> None:
        assert sys.version_info >= (3, 14)
        assert not sys._is_gil_enabled()

    def test_the_gil_is_pinned_off_rather_than_merely_off(self) -> None:
        """Unpinned, the first undeclared extension module switches it back on mid-run."""
        assert sys.flags.gil == 0

    def test_the_undeclared_extensions_we_depend_on_do_not_switch_it_back_on(self) -> None:
        """lxml and SQLAlchemy's cyextension are exactly the ones that would."""
        probe = (
            "import docx, sqlalchemy, sys; "
            "print(sys._is_gil_enabled())"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, cwd=ROOT
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "False"

    def test_importing_the_package_is_refused_when_the_gil_is_not_pinned(self) -> None:
        """The tripwire: a run that would silently re-acquire the GIL never starts."""
        result = subprocess.run(
            [sys.executable, "-c", "import system"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env={"PATH": "/usr/bin:/bin"},
        )
        assert result.returncode != 0
        assert "PYTHON_GIL=0" in result.stderr

    def test_lxml_is_the_one_that_re_enables_the_gil(self) -> None:
        """Named exactly, because it is the constraint `extract_many` is built around.

        Run unpinned, so the switch is observable. This is what keeps the extraction
        pool off the DOCX, XLSX and email backends.
        """
        result = subprocess.run(
            [sys.executable, "-c", "import lxml.etree, sys; print(sys._is_gil_enabled())"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env={"PATH": "/usr/bin:/bin"},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "True", "if lxml stopped doing this, ADR 0006 shrinks"

    def test_the_checkpoint_serializer_keeps_the_gil_off(self) -> None:
        """LangGraph checkpoints go through ormsgpack, not orjson.

        ormsgpack is a native extension on the durability path that ADR 0006 never
        inventoried, because the orjson stand-in was assumed to be carrying checkpoints.
        It is not — `langgraph.checkpoint.serde.jsonplus` packs through ormsgpack, and a
        full offline run calls `orjson.dumps` zero times. What matters about ormsgpack,
        then, is not its behaviour but that it declares free-threaded support: unpinned,
        importing it leaves the GIL off.
        """
        result = subprocess.run(
            [sys.executable, "-c", "import ormsgpack, sys; print(sys._is_gil_enabled())"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env={"PATH": "/usr/bin:/bin"},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "False"


class TestTheOrjsonStandIn:
    """LangGraph, its SDK and LangSmith serialize through these exact behaviours."""

    def test_dumps_returns_compact_utf8_bytes(self) -> None:
        assert orjson.dumps({"a": 1, "b": [1, 2]}) == b'{"a":1,"b":[1,2]}'
        assert orjson.dumps("é") == '"é"'.encode()

    def test_loads_accepts_bytes_and_str_alike(self) -> None:
        assert orjson.loads(b'{"a":1}') == {"a": 1}
        assert orjson.loads('{"a":1}') == {"a": 1}
        assert orjson.loads(memoryview(b"[1]")) == [1]

    def test_a_malformed_document_raises_the_decode_error_callers_catch(self) -> None:
        with pytest.raises(orjson.JSONDecodeError):
            orjson.loads(b"{not json")
        # LangSmith catches `json.JSONDecodeError`; upstream orjson subclasses it.
        assert issubclass(orjson.JSONDecodeError, json.JSONDecodeError)
        assert issubclass(orjson.JSONDecodeError, ValueError)

    def test_a_fragment_is_spliced_in_already_serialized(self) -> None:
        """LangGraph embeds pre-serialized channel values without re-encoding them."""
        payload = orjson.dumps({"v": orjson.Fragment(b'{"already":"json"}')})
        assert payload == b'{"v":{"already":"json"}}'

    def test_several_fragments_keep_their_own_positions(self) -> None:
        payload = orjson.dumps(
            [orjson.Fragment('{"n":1}'), orjson.Fragment('{"n":2}')]
        )
        assert payload == b'[{"n":1},{"n":2}]'

    def test_a_string_that_looks_like_the_placeholder_is_not_substituted(self) -> None:
        """The nonce is what keeps payload data from being mistaken for a fragment."""
        payload = orjson.dumps({"a": "\x00orjson-0000:0", "b": orjson.Fragment("1")})
        assert orjson.loads(payload) == {"a": "\x00orjson-0000:0", "b": 1}

    def test_non_finite_floats_serialize_as_null_not_as_nan(self) -> None:
        """`NaN` is not JSON, and a checkpoint containing it would not load back."""
        assert orjson.dumps([math.nan, math.inf, -math.inf]) == b"[null,null,null]"

    def test_a_non_string_key_is_refused_unless_the_option_allows_it(self) -> None:
        with pytest.raises(TypeError):
            orjson.dumps({1: "a"})
        assert orjson.dumps({1: "a"}, option=orjson.OPT_NON_STR_KEYS) == b'{"1":"a"}'

    def test_the_types_orjson_serializes_natively_do_not_reach_default(self) -> None:
        """`default` raising on a UUID is normal upstream, because it never sees one."""

        def default(value: object) -> object:
            raise AssertionError(f"default saw {value!r}")

        moment = datetime.datetime(2026, 8, 28, 12, 0, tzinfo=datetime.UTC)
        identifier = uuid.UUID("00000000-0000-4000-8000-000000000000")
        assert orjson.dumps(moment, default=default) == b'"2026-08-28T12:00:00+00:00"'
        assert orjson.dumps(identifier, default=default) == f'"{identifier}"'.encode()
        assert orjson.dumps(datetime.date(2026, 8, 28), default=default) == b'"2026-08-28"'

    def test_dataclasses_and_enums_serialize_by_value(self) -> None:
        class Colour(enum.Enum):
            RED = "red"

        @dataclasses.dataclass
        class Item:
            name: str
            colour: Colour

        assert orjson.dumps(Item("a", Colour.RED)) == b'{"name":"a","colour":"red"}'

    def test_an_unknown_type_goes_to_default_and_its_result_is_serialized(self) -> None:
        class Opaque:
            pass

        assert orjson.dumps(Opaque(), default=lambda _: {"k": (1, 2)}) == b'{"k":[1,2]}'

    def test_an_unknown_type_without_a_default_raises_a_type_error(self) -> None:
        with pytest.raises(TypeError):
            orjson.dumps(object())

    def test_the_options_callers_pass_are_honoured(self) -> None:
        assert orjson.dumps({"b": 1, "a": 2}, option=orjson.OPT_SORT_KEYS) == b'{"a":2,"b":1}'
        assert orjson.dumps({"a": 1}, option=orjson.OPT_INDENT_2) == b'{\n  "a": 1\n}'
        assert orjson.dumps([1], option=orjson.OPT_APPEND_NEWLINE) == b"[1]\n"

    def test_numpy_values_serialize_only_under_their_option(self) -> None:
        numpy = pytest.importorskip("numpy")
        array = numpy.array([1, 2])
        assert orjson.dumps(array, option=orjson.OPT_SERIALIZE_NUMPY) == b"[1,2]"
        with pytest.raises(TypeError):
            orjson.dumps(array)

    def test_an_integer_beyond_64_bits_is_refused(self) -> None:
        with pytest.raises(TypeError):
            orjson.dumps(2**64)

    def test_the_option_constants_hold_orjsons_own_values(self) -> None:
        """Callers or them together, so the numbers themselves are part of the contract."""
        assert (orjson.OPT_INDENT_2, orjson.OPT_NON_STR_KEYS) == (1, 4)
        assert (orjson.OPT_SERIALIZE_NUMPY, orjson.OPT_SORT_KEYS) == (128, 256)
        assert orjson.OPT_APPEND_NEWLINE == 2048


class TestTheFastuuidStandIn:
    def test_it_hands_back_the_standard_librarys_uuids(self) -> None:
        value = fastuuid.uuid4()
        assert isinstance(value, uuid.UUID)
        assert value.version == 4
        assert fastuuid.uuid4() != value

    def test_the_bulk_helpers_keep_the_surface_complete(self) -> None:
        assert len(set(fastuuid.uuid4_bulk(3))) == 3
        assert all(isinstance(v, str) for v in fastuuid.uuid4_as_strings_bulk(2))


class TestTokenCounting:
    """True of either install: LiteLLM imports, and OpenAI-shaped models use tiktoken."""

    def test_litellm_imports_which_is_what_all_of_this_was_for(self) -> None:
        import litellm

        assert callable(litellm.completion)

    def test_an_openai_shaped_model_is_counted_by_tiktoken(self) -> None:
        """`SHAKESPEARE_MODEL` pins an OpenAI-shaped model, so this is the path we use."""
        from litellm import token_counter

        assert token_counter(model="gpt-4o", text="hello there") > 0


@needs_the_stand_in
class TestTheTokenizersStandIn:
    def test_asking_for_a_tokenizer_fails_loudly_rather_than_approximating(self) -> None:
        """A guessed token count would be a wrong bill and a wrong batch size.

        This is the stand-in's own contract. What LiteLLM does with it is the next test.
        """
        with pytest.raises(tokenizers.TokenizerUnavailable):
            tokenizers.Tokenizer.from_pretrained("Xenova/llama-3-tokenizer")
        with pytest.raises(tokenizers.TokenizerUnavailable):
            tokenizers.Tokenizer.from_str("{}")

    def test_a_claude_model_is_counted_by_tiktoken_because_litellm_swallows_the_refusal(
        self,
    ) -> None:
        """The honest record of what actually happens, not what the stand-in intends.

        `litellm.utils._select_tokenizer_helper` wraps tokenizer selection in a bare
        `except Exception`, logs at debug and falls back to tiktoken. So the refusal never
        reaches a caller: a Claude model is counted by the wrong tokenizer and nothing
        says so. This test exists to fail the day that changes — in either direction.
        """
        from litellm import token_counter

        assert token_counter(model="claude-sonnet-4-5", text="hello there friend") > 0


class TestTheVendoredTokenizer:
    """The wheel HuggingFace does not publish, built by `vendor/build-tokenizers.sh`."""

    @needs_the_real_one
    def test_it_tokenizes_rather_than_refusing(self) -> None:
        encoding = tokenizers.Tokenizer.from_str(_claude_tokenizer_json()).encode("hello there")
        assert len(encoding.ids) == 2

    @needs_the_real_one
    def test_importing_it_does_not_switch_the_gil_back_on(self) -> None:
        """Its Rust declares `gil_used = false`, which is why the wheel was worth building."""
        result = subprocess.run(
            [sys.executable, "-c", "import tokenizers, sys; print(sys._is_gil_enabled())"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            env={"PATH": "/usr/bin:/bin"},  # unpinned: nothing is forcing the GIL off
        )
        assert result.stdout.strip() == "False", result.stderr

    @needs_the_real_one
    def test_litellm_counts_claude_with_claudes_own_tokenizer(self) -> None:
        """The point of the exercise: an exact count, not a tiktoken approximation."""
        from litellm.utils import _select_tokenizer_helper

        assert _select_tokenizer_helper("claude-sonnet-4-5")["type"] == "huggingface_tokenizer"


def _claude_tokenizer_json() -> str:
    """LiteLLM bundles it, so this needs no network."""
    from importlib import resources

    return (
        resources.files("litellm.litellm_core_utils.tokenizers")
        .joinpath("anthropic_tokenizer.json")
        .read_text()
    )


class TestTheGatewayGate:
    """`profile_from_environment` refuses only what this install cannot count."""

    def _profile(self, monkeypatch: pytest.MonkeyPatch, model: str) -> None:
        from system.model_access import profile_from_environment

        monkeypatch.setenv("SHAKESPEARE_MODEL", model)
        profile_from_environment()

    def test_an_openai_shaped_model_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._profile(monkeypatch, "openrouter/openai/gpt-5-mini")

    def test_a_claude_3_model_is_accepted_because_litellm_counts_it_with_tiktoken(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gate tracks LiteLLM's own choice, not the vendor's name."""
        self._profile(monkeypatch, "claude-3-5-sonnet-20240620")

    @needs_the_stand_in
    @pytest.mark.parametrize(
        "model", ["claude-sonnet-4-5", "meta-llama/llama-3-8b", "replicate/meta/llama-2-70b"]
    )
    def test_a_model_it_cannot_count_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, model: str
    ) -> None:
        """Counting these with tiktoken is a wrong bill, so the run does not start."""
        from system.contracts import ErrorCode
        from system.model_access import GatewayError

        with pytest.raises(GatewayError) as raised:
            self._profile(monkeypatch, model)
        assert raised.value.code is ErrorCode.MODEL_PERMANENT
        assert "openrouter/openai" in str(raised.value)

    @needs_the_real_one
    @pytest.mark.parametrize(
        "model", ["claude-sonnet-4-5", "meta-llama/llama-3-8b", "replicate/meta/llama-2-70b"]
    )
    def test_nothing_is_refused_once_the_real_tokenizer_is_installed(
        self, monkeypatch: pytest.MonkeyPatch, model: str
    ) -> None:
        """The gate lifts itself: it exists only where a count would have to be guessed."""
        self._profile(monkeypatch, model)

    def test_the_private_litellm_helper_the_gate_reads_still_exists(self) -> None:
        """The coupling that would fail open: no helper, no gate, silent mis-counting."""
        from litellm.utils import _return_huggingface_tokenizer

        assert callable(_return_huggingface_tokenizer)
        assert _return_huggingface_tokenizer("gpt-4o") is None
