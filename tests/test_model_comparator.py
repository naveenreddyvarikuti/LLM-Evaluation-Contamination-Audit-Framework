from llm_eval_audit.analysis.model_comparator import ModelComparator
from llm_eval_audit.core.types import PairwiseComparison, Verdict


def _comparisons(a_wins, b_wins, ties=0):
    comps = []
    for i in range(a_wins):
        comps.append(PairwiseComparison(example_id=f"a{i}", model_a="A", model_b="B", verdict=Verdict.A))
    for i in range(b_wins):
        comps.append(PairwiseComparison(example_id=f"b{i}", model_a="A", model_b="B", verdict=Verdict.B))
    for i in range(ties):
        comps.append(PairwiseComparison(example_id=f"t{i}", model_a="A", model_b="B", verdict=Verdict.TIE))
    return comps


def test_dominant_model_gets_higher_elo():
    comparator = ModelComparator(bootstrap_iterations=200, random_seed=42)
    comparisons = _comparisons(a_wins=18, b_wins=2)
    ratings = comparator.bootstrap_elo_ci(comparisons)
    assert ratings["A"].rating > ratings["B"].rating


def test_win_rates_sum_to_one_per_matchup():
    comparisons = _comparisons(a_wins=7, b_wins=3)
    win_rates = ModelComparator.win_rates(comparisons)
    assert abs(win_rates["A"] + win_rates["B"] - 1.0) < 1e-9
    assert win_rates["A"] == 0.7


def test_ties_split_win_rate_evenly():
    comparisons = _comparisons(a_wins=0, b_wins=0, ties=4)
    win_rates = ModelComparator.win_rates(comparisons)
    assert win_rates["A"] == 0.5
    assert win_rates["B"] == 0.5


def test_equal_strength_models_not_distinguishable():
    comparator = ModelComparator(bootstrap_iterations=200, random_seed=1)
    comparisons = _comparisons(a_wins=5, b_wins=5)
    ratings = comparator.bootstrap_elo_ci(comparisons)
    assert not ModelComparator.is_distinguishable(ratings["A"], ratings["B"])
