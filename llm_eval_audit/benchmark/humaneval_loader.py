"""Loads the public HumanEval benchmark (openai_humaneval) as `Example` objects.

HumanEval is deliberately used as the "known contaminated" public benchmark:
it has been circulating on GitHub and in scraped web text since 2021, so
most modern code models have likely seen it (or close paraphrases) during
pretraining. Running the n-gram contamination detector against it is a
good demonstration that the detector actually flags something real, as
opposed to only ever seeing clean private examples.
"""

from __future__ import annotations

from llm_eval_audit.core.logging_config import get_logger
from llm_eval_audit.core.types import CapabilityCategory, Difficulty, Example

logger = get_logger(__name__)


def load_humaneval(limit: int | None = None) -> list[Example]:
    """Load HumanEval problems from HuggingFace `datasets` as `Example` objects.

    `test_cases` holds the HumanEval `test` field verbatim (it defines a
    `check(candidate)` function), and `entry_point` is passed straight
    through so `PassAtKEvaluator` can call `check(<entry_point>)` after
    executing prompt + completion.
    """
    from datasets import load_dataset  # deferred import: only needed when actually loading

    dataset = load_dataset("openai_humaneval", split="test")
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    examples = [
        Example(
            id=record["task_id"],
            question=record["prompt"],
            reference_answer=record["canonical_solution"],
            capability_category=CapabilityCategory.CODE_GENERATION,
            difficulty=Difficulty.MEDIUM,
            metadata={"source": "openai_humaneval"},
            test_cases=[record["test"]],
            entry_point=record["entry_point"],
        )
        for record in dataset
    ]
    logger.info("Loaded %d HumanEval problems", len(examples))
    return examples
