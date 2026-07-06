"""Benchmark Manager: stores, versions, and manages private evaluation examples.

Each benchmark version is a JSONL file of `Example` records plus a sidecar
JSON file of canary strings that were injected into (or reserved for) that
version. Versions are immutable once written — "retiring" an example sets a
`retired` flag rather than deleting it, so historical eval runs that pinned
a version stay reproducible.
"""

from __future__ import annotations

import json
import secrets
import string
from pathlib import Path
from typing import Optional

from llm_eval_audit.core.logging_config import get_logger
from llm_eval_audit.core.types import BenchmarkVersion, CapabilityCategory, Difficulty, Example

logger = get_logger(__name__)


class BenchmarkManager:
    """Manages versioned benchmark files stored as JSONL on disk.

    Layout on disk (relative to `base_path`):
        {base_path}/{version}.jsonl              - one Example per line
        {base_path}/../canaries/{version}.json   - list[str] canary strings
    """

    def __init__(self, base_path: str | Path = "data/benchmarks", canary_path: Optional[str | Path] = None):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.canary_path = Path(canary_path) if canary_path else self.base_path.parent / "canaries"
        self.canary_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------
    def _jsonl_path(self, version: str) -> Path:
        return self.base_path / f"{version}.jsonl"

    def _canary_file_path(self, version: str) -> Path:
        return self.canary_path / f"{version}.json"

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------
    def load_version(self, version: str) -> BenchmarkVersion:
        """Load a benchmark version (examples + canaries) from disk.

        Raises FileNotFoundError if the version does not exist yet — callers
        creating a brand new version should catch this and start with
        `BenchmarkVersion(version=version, examples=[])` instead.
        """
        jsonl_path = self._jsonl_path(version)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Benchmark version '{version}' not found at {jsonl_path}")

        examples: list[Example] = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    examples.append(Example.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    raise ValueError(
                        f"Malformed example on line {line_num} of {jsonl_path}: {e}"
                    ) from e

        canaries = self._load_canaries(version)
        logger.info("Loaded benchmark version '%s': %d examples, %d canaries", version, len(examples), len(canaries))
        return BenchmarkVersion(version=version, examples=examples, canary_strings=canaries)

    def _load_canaries(self, version: str) -> list[str]:
        canary_file = self._canary_file_path(version)
        if not canary_file.exists():
            return []
        with canary_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_canaries(self, version: str, canaries: list[str]) -> None:
        with self._canary_file_path(version).open("w", encoding="utf-8") as f:
            json.dump(canaries, f, indent=2)

    def save_version(self, bench: BenchmarkVersion) -> None:
        """Persist a BenchmarkVersion's examples and canaries to disk."""
        jsonl_path = self._jsonl_path(bench.version)
        with jsonl_path.open("w", encoding="utf-8") as f:
            for example in bench.examples:
                f.write(json.dumps(example.to_dict()) + "\n")
        self._save_canaries(bench.version, bench.canary_strings)
        logger.info(
            "Saved benchmark version '%s': %d examples, %d canaries",
            bench.version, len(bench.examples), len(bench.canary_strings),
        )

    def version_exists(self, version: str) -> bool:
        return self._jsonl_path(version).exists()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def add_example(
        self,
        version: str,
        example: Example,
        inject_canary: bool = False,
    ) -> Example:
        """Add a new example to a benchmark version, creating the version if needed.

        If `inject_canary` is True, a fresh canary string is generated and
        appended to `example.metadata["canary"]` as well as the version's
        canary registry, so contamination checks later can test whether this
        exact example (and only this example) leaked into training data.
        """
        try:
            bench = self.load_version(version)
        except FileNotFoundError:
            bench = BenchmarkVersion(version=version, examples=[])

        if any(e.id == example.id for e in bench.examples):
            raise ValueError(f"Example id '{example.id}' already exists in version '{version}'")

        if inject_canary:
            canary = self.generate_canary_string()
            example.canary = canary
            bench.canary_strings.append(canary)

        bench.examples.append(example)
        self.save_version(bench)
        return example

    def retire_example(self, version: str, example_id: str) -> None:
        """Mark an example as retired (soft delete) within a version.

        Retiring, rather than deleting, preserves reproducibility: an eval
        run that recorded "benchmark v1, N examples" won't silently change
        shape, and `BenchmarkVersion.active_examples()` filters retired
        examples out for anything downstream.
        """
        bench = self.load_version(version)
        for e in bench.examples:
            if e.id == example_id:
                e.retired = True
                self.save_version(bench)
                logger.info("Retired example '%s' in version '%s'", example_id, version)
                return
        raise KeyError(f"Example id '{example_id}' not found in version '{version}'")

    # ------------------------------------------------------------------
    # Canary strings
    # ------------------------------------------------------------------
    @staticmethod
    def generate_canary_string(length: int = 32, prefix: str = "CANARY") -> str:
        """Generate a unique, unguessable canary string.

        Uses `secrets` (not `random`) so canaries can't be predicted, and a
        recognizable prefix so a regex/substring search over generated text
        can find them even without the exact registry.
        """
        alphabet = string.ascii_letters + string.digits
        token = "".join(secrets.choice(alphabet) for _ in range(length))
        return f"{prefix}-{token}"

    def get_canaries(self, version: str) -> list[str]:
        return self._load_canaries(version)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_to_jsonl(
        self,
        version: str,
        output_path: str | Path,
        include_retired: bool = False,
    ) -> Path:
        """Export a benchmark version to a standalone JSONL file.

        Useful for handing a clean, portable snapshot of the benchmark to
        another tool (e.g. an eval harness that only understands JSONL).
        """
        bench = self.load_version(version)
        examples = bench.examples if include_retired else bench.active_examples()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example.to_dict()) + "\n")
        logger.info("Exported %d examples from version '%s' to %s", len(examples), version, output_path)
        return output_path

    def list_versions(self) -> list[str]:
        return sorted(p.stem for p in self.base_path.glob("*.jsonl"))
