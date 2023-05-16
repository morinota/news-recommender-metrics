import math

from news_recommender_metrics.utils.probability_mass_function.rank_aware_probability_mass_function import (
    RankAwareProbabilityMassFunction,
)


def calc_rank_weight_MMR(rank: int) -> float:
    return 1 / rank


def test_calc_rank_aware_pmf() -> None:
    R = ["a", "b", "b"]
    Q_asterisk_expected = {
        "a": calc_rank_weight_MMR(1)
        / (calc_rank_weight_MMR(1) + calc_rank_weight_MMR(2) + calc_rank_weight_MMR(3)),
        "b": (calc_rank_weight_MMR(2) + calc_rank_weight_MMR(3))
        / (calc_rank_weight_MMR(1) + calc_rank_weight_MMR(2) + calc_rank_weight_MMR(3)),
    }
    print(Q_asterisk_expected)

    Q_asterisk_actual = RankAwareProbabilityMassFunction.from_ranking(R)
    assert [
        math.isclose(
            p_x_actual,
            Q_asterisk_expected.get(x, 0),
            rel_tol=1e-3,
        )
        for x, p_x_actual in Q_asterisk_actual.pmf.items()
    ]
    assert sum(Q_asterisk_actual.pmf.values()) == 1.0
