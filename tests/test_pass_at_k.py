import pytest

from llm_eval_audit.evaluation.pass_at_k import unbiased_pass_at_k


def test_pass_at_k_all_correct():
    assert unbiased_pass_at_k(n=10, c=10, k=1) == 1.0


def test_pass_at_k_all_wrong():
    assert unbiased_pass_at_k(n=10, c=0, k=1) == 0.0


def test_pass_at_k_matches_naive_when_k_equals_n():
    # If k == n, pass@k is 1.0 iff at least one sample passed.
    assert unbiased_pass_at_k(n=5, c=1, k=5) == 1.0
    assert unbiased_pass_at_k(n=5, c=0, k=5) == 0.0


def test_pass_at_k_monotonic_in_k():
    n, c = 200, 40
    values = [unbiased_pass_at_k(n, c, k) for k in (1, 5, 10, 100)]
    assert values == sorted(values)


def test_pass_at_k_raises_when_k_greater_than_n():
    with pytest.raises(ValueError):
        unbiased_pass_at_k(n=5, c=1, k=10)
