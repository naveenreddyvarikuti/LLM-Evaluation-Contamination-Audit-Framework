"""Seeds benchmark version "v1" with 20 private examples across 4 capability
categories (function completion, bug fixing, code explanation, refactoring),
each with an injected canary string for contamination tracking.

Run once: `python scripts/seed_benchmark.py`
Re-running is safe-ish but will raise on duplicate ids; delete
`data/benchmarks/v1.jsonl` and `data/canaries/v1.json` first if you want to
regenerate from scratch.
"""

from __future__ import annotations

from llm_eval_audit.benchmark.manager import BenchmarkManager
from llm_eval_audit.core.types import CapabilityCategory, Difficulty, Example

VERSION = "v1"

PRIVATE_EXAMPLES: list[Example] = [
    # --- Function completion (5, easy-medium) ---
    Example(
        id="fc-01",
        question="Complete the function:\n\ndef is_palindrome(s: str) -> bool:\n    \"\"\"Return True if s reads the same forwards and backwards, ignoring case.\"\"\"\n",
        reference_answer="    s = s.lower()\n    return s == s[::-1]",
        capability_category=CapabilityCategory.FUNCTION_COMPLETION,
        difficulty=Difficulty.EASY,
    ),
    Example(
        id="fc-02",
        question="Complete the function:\n\ndef flatten(nested: list) -> list:\n    \"\"\"Flatten an arbitrarily nested list of lists into a single flat list.\"\"\"\n",
        reference_answer=(
            "    result = []\n"
            "    for item in nested:\n"
            "        if isinstance(item, list):\n"
            "            result.extend(flatten(item))\n"
            "        else:\n"
            "            result.append(item)\n"
            "    return result"
        ),
        capability_category=CapabilityCategory.FUNCTION_COMPLETION,
        difficulty=Difficulty.MEDIUM,
    ),
    Example(
        id="fc-03",
        question="Complete the function:\n\ndef most_common_word(text: str) -> str:\n    \"\"\"Return the most frequently occurring word in text (case-insensitive).\"\"\"\n",
        reference_answer=(
            "    from collections import Counter\n"
            "    words = text.lower().split()\n"
            "    return Counter(words).most_common(1)[0][0]"
        ),
        capability_category=CapabilityCategory.FUNCTION_COMPLETION,
        difficulty=Difficulty.MEDIUM,
    ),
    Example(
        id="fc-04",
        question="Complete the function:\n\ndef chunk_list(items: list, size: int) -> list:\n    \"\"\"Split items into consecutive chunks of at most `size` elements each.\"\"\"\n",
        reference_answer="    return [items[i:i + size] for i in range(0, len(items), size)]",
        capability_category=CapabilityCategory.FUNCTION_COMPLETION,
        difficulty=Difficulty.EASY,
    ),
    Example(
        id="fc-05",
        question="Complete the function:\n\ndef merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:\n    \"\"\"Merge overlapping (start, end) intervals and return them sorted.\"\"\"\n",
        reference_answer=(
            "    if not intervals:\n"
            "        return []\n"
            "    intervals = sorted(intervals)\n"
            "    merged = [intervals[0]]\n"
            "    for start, end in intervals[1:]:\n"
            "        last_start, last_end = merged[-1]\n"
            "        if start <= last_end:\n"
            "            merged[-1] = (last_start, max(last_end, end))\n"
            "        else:\n"
            "            merged.append((start, end))\n"
            "    return merged"
        ),
        capability_category=CapabilityCategory.FUNCTION_COMPLETION,
        difficulty=Difficulty.MEDIUM,
    ),
    # --- Bug fixing (5, medium-hard) ---
    Example(
        id="bf-01",
        question=(
            "Fix the bug in this function, which should return the average of a list "
            "but crashes on an empty list:\n\n"
            "def average(nums):\n    return sum(nums) / len(nums)\n"
        ),
        reference_answer="def average(nums):\n    if not nums:\n        return 0.0\n    return sum(nums) / len(nums)",
        capability_category=CapabilityCategory.BUG_FIXING,
        difficulty=Difficulty.MEDIUM,
    ),
    Example(
        id="bf-02",
        question=(
            "Fix the off-by-one bug: this should return the last `n` items of a list, "
            "but currently returns one extra item.\n\n"
            "def last_n(items, n):\n    return items[-n-1:]\n"
        ),
        reference_answer="def last_n(items, n):\n    return items[-n:] if n > 0 else []",
        capability_category=CapabilityCategory.BUG_FIXING,
        difficulty=Difficulty.MEDIUM,
    ),
    Example(
        id="bf-03",
        question=(
            "This function is supposed to deduplicate a list while preserving order, "
            "but it uses a set and loses ordering. Fix it.\n\n"
            "def dedupe(items):\n    return list(set(items))\n"
        ),
        reference_answer=(
            "def dedupe(items):\n"
            "    seen = set()\n"
            "    result = []\n"
            "    for item in items:\n"
            "        if item not in seen:\n"
            "            seen.add(item)\n"
            "            result.append(item)\n"
            "    return result"
        ),
        capability_category=CapabilityCategory.BUG_FIXING,
        difficulty=Difficulty.HARD,
    ),
    Example(
        id="bf-04",
        question=(
            "This recursive factorial function causes infinite recursion for n=0. Fix the base case.\n\n"
            "def factorial(n):\n    return n * factorial(n - 1)\n"
        ),
        reference_answer="def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
        capability_category=CapabilityCategory.BUG_FIXING,
        difficulty=Difficulty.MEDIUM,
    ),
    Example(
        id="bf-05",
        question=(
            "This function mutates the caller's default argument across calls (classic Python "
            "mutable default argument bug). Fix it.\n\n"
            "def append_item(item, items=[]):\n    items.append(item)\n    return items\n"
        ),
        reference_answer=(
            "def append_item(item, items=None):\n"
            "    if items is None:\n"
            "        items = []\n"
            "    items.append(item)\n"
            "    return items"
        ),
        capability_category=CapabilityCategory.BUG_FIXING,
        difficulty=Difficulty.HARD,
    ),
    # --- Code explanation (5, medium) ---
    Example(
        id="ce-01",
        question="Explain what this function does and why it works:\n\ndef f(n):\n    return n & (n - 1) == 0 and n != 0",
        reference_answer=(
            "It checks whether n is a power of two. `n - 1` flips all bits after the lowest "
            "set bit of n; ANDing with n clears that lowest set bit. A power of two has exactly "
            "one set bit, so the result is 0 only for powers of two (excluding 0, handled by the "
            "explicit n != 0 check)."
        ),
        capability_category=CapabilityCategory.CODE_EXPLANATION,
        difficulty=Difficulty.MEDIUM,
    ),
    Example(
        id="ce-02",
        question="Explain what this decorator does:\n\ndef memoize(fn):\n    cache = {}\n    def wrapper(*args):\n        if args not in cache:\n            cache[args] = fn(*args)\n        return cache[args]\n    return wrapper",
        reference_answer=(
            "It's a memoization decorator: it wraps `fn` so that results are cached by their "
            "positional argument tuple. On repeated calls with the same arguments, it returns the "
            "cached result instead of recomputing `fn`, trading memory for time."
        ),
        capability_category=CapabilityCategory.CODE_EXPLANATION,
        difficulty=Difficulty.MEDIUM,
    ),
    Example(
        id="ce-03",
        question="Explain the time complexity of this function and why:\n\ndef contains_duplicate(nums):\n    seen = set()\n    for n in nums:\n        if n in seen:\n            return True\n        seen.add(n)\n    return False",
        reference_answer=(
            "O(n) time and O(n) space. Each element is checked against and inserted into a hash "
            "set once, and both operations are O(1) average case, giving a single linear pass "
            "over the input."
        ),
        capability_category=CapabilityCategory.CODE_EXPLANATION,
        difficulty=Difficulty.MEDIUM,
    ),
    Example(
        id="ce-04",
        question="Explain what this generator function does:\n\ndef sliding_window(seq, size):\n    for i in range(len(seq) - size + 1):\n        yield seq[i:i + size]",
        reference_answer=(
            "It lazily yields every contiguous sub-sequence ('window') of length `size` from "
            "`seq`, sliding one position at a time from the start to the last valid starting index, "
            "without materializing all windows in memory at once."
        ),
        capability_category=CapabilityCategory.CODE_EXPLANATION,
        difficulty=Difficulty.MEDIUM,
    ),
    Example(
        id="ce-05",
        question="Explain why this function is not thread-safe:\n\ncounter = 0\ndef increment():\n    global counter\n    counter += 1",
        reference_answer=(
            "`counter += 1` is not atomic — it's a read, an increment, and a write, executed as "
            "separate bytecode steps. If two threads interleave between the read and the write, "
            "one thread's increment can be lost, so concurrent calls can undercount the total."
        ),
        capability_category=CapabilityCategory.CODE_EXPLANATION,
        difficulty=Difficulty.MEDIUM,
    ),
    # --- Refactoring (5, medium-hard) ---
    Example(
        id="rf-01",
        question=(
            "Refactor this function to remove the repeated `if/elif` chain using a dictionary "
            "dispatch instead:\n\n"
            "def apply_op(op, a, b):\n"
            "    if op == 'add':\n        return a + b\n"
            "    elif op == 'sub':\n        return a - b\n"
            "    elif op == 'mul':\n        return a * b\n"
            "    elif op == 'div':\n        return a / b\n"
        ),
        reference_answer=(
            "import operator\n\n"
            "_OPS = {'add': operator.add, 'sub': operator.sub, 'mul': operator.mul, 'div': operator.truediv}\n\n"
            "def apply_op(op, a, b):\n    return _OPS[op](a, b)"
        ),
        capability_category=CapabilityCategory.REFACTORING,
        difficulty=Difficulty.MEDIUM,
    ),
    Example(
        id="rf-02",
        question=(
            "Refactor this nested-loop duplicate-pair finder into a single pass using a set, "
            "improving it from O(n^2) to O(n):\n\n"
            "def has_pair_with_sum(nums, target):\n"
            "    for i in range(len(nums)):\n"
            "        for j in range(i + 1, len(nums)):\n"
            "            if nums[i] + nums[j] == target:\n"
            "                return True\n"
            "    return False\n"
        ),
        reference_answer=(
            "def has_pair_with_sum(nums, target):\n"
            "    seen = set()\n"
            "    for n in nums:\n"
            "        if target - n in seen:\n"
            "            return True\n"
            "        seen.add(n)\n"
            "    return False"
        ),
        capability_category=CapabilityCategory.REFACTORING,
        difficulty=Difficulty.HARD,
    ),
    Example(
        id="rf-03",
        question=(
            "Refactor this class to use a dataclass instead of a hand-written __init__/__repr__:\n\n"
            "class Point:\n"
            "    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n"
            "    def __repr__(self):\n        return f'Point({self.x}, {self.y})'\n"
        ),
        reference_answer=(
            "from dataclasses import dataclass\n\n"
            "@dataclass\n"
            "class Point:\n    x: float\n    y: float"
        ),
        capability_category=CapabilityCategory.REFACTORING,
        difficulty=Difficulty.MEDIUM,
    ),
    Example(
        id="rf-04",
        question=(
            "Refactor this function to use a list comprehension instead of the manual loop:\n\n"
            "def squares_of_evens(nums):\n"
            "    result = []\n"
            "    for n in nums:\n        if n % 2 == 0:\n            result.append(n * n)\n"
            "    return result\n"
        ),
        reference_answer="def squares_of_evens(nums):\n    return [n * n for n in nums if n % 2 == 0]",
        capability_category=CapabilityCategory.REFACTORING,
        difficulty=Difficulty.MEDIUM,
    ),
    Example(
        id="rf-05",
        question=(
            "Refactor this function to use a context manager instead of manual open/close, "
            "ensuring the file is closed even if reading raises:\n\n"
            "def read_file(path):\n"
            "    f = open(path)\n    data = f.read()\n    f.close()\n    return data\n"
        ),
        reference_answer="def read_file(path):\n    with open(path) as f:\n        return f.read()",
        capability_category=CapabilityCategory.REFACTORING,
        difficulty=Difficulty.HARD,
    ),
]


def main() -> None:
    manager = BenchmarkManager()
    if manager.version_exists(VERSION):
        print(f"Version '{VERSION}' already exists; skipping seed. "
              f"Delete data/benchmarks/{VERSION}.jsonl to reseed.")
        return

    for example in PRIVATE_EXAMPLES:
        manager.add_example(VERSION, example, inject_canary=True)

    bench = manager.load_version(VERSION)
    print(f"Seeded benchmark '{VERSION}' with {len(bench.examples)} examples "
          f"and {len(bench.canary_strings)} canary strings.")


if __name__ == "__main__":
    main()
