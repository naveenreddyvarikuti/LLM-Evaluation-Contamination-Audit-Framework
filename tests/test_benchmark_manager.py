import pytest

from llm_eval_audit.benchmark.manager import BenchmarkManager
from llm_eval_audit.core.types import CapabilityCategory, Difficulty, Example


@pytest.fixture
def manager(tmp_path):
    return BenchmarkManager(base_path=tmp_path / "benchmarks", canary_path=tmp_path / "canaries")


def _example(example_id="e1"):
    return Example(
        id=example_id,
        question="q",
        reference_answer="a",
        capability_category=CapabilityCategory.FUNCTION_COMPLETION,
        difficulty=Difficulty.EASY,
    )


def test_add_and_load_example(manager):
    manager.add_example("v1", _example())
    bench = manager.load_version("v1")
    assert len(bench.examples) == 1
    assert bench.examples[0].id == "e1"


def test_add_example_with_canary_registers_it(manager):
    manager.add_example("v1", _example(), inject_canary=True)
    bench = manager.load_version("v1")
    assert bench.examples[0].canary is not None
    assert bench.examples[0].canary in bench.canary_strings


def test_duplicate_id_raises(manager):
    manager.add_example("v1", _example())
    with pytest.raises(ValueError):
        manager.add_example("v1", _example())


def test_retire_example_excludes_from_active(manager):
    manager.add_example("v1", _example("e1"))
    manager.add_example("v1", _example("e2"))
    manager.retire_example("v1", "e1")

    bench = manager.load_version("v1")
    active_ids = {e.id for e in bench.active_examples()}
    assert active_ids == {"e2"}
    assert len(bench.examples) == 2  # retired example still present, just flagged


def test_retire_missing_example_raises(manager):
    manager.add_example("v1", _example())
    with pytest.raises(KeyError):
        manager.retire_example("v1", "does-not-exist")


def test_export_to_jsonl_excludes_retired_by_default(manager, tmp_path):
    manager.add_example("v1", _example("e1"))
    manager.add_example("v1", _example("e2"))
    manager.retire_example("v1", "e1")

    out_path = manager.export_to_jsonl("v1", tmp_path / "export.jsonl")
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
