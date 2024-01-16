from typing import Any

from news_recommender_metrics.RADio.divergence_metrics import JSDivergence
from news_recommender_metrics.utils.probability_mass_function.probability_mass_function import (
    ProbabilityMassFunction,
)
from news_recommender_metrics.utils.probability_mass_function.rank_aware_probability_mass_function import (
    RankAwareProbabilityMassFunction,
)


class Calibration:
    def __init__(
        self,
        is_rank_aware: bool = True,
        rank_weight_method: str = "MMR",
    ) -> None:
        self.is_rank_aware = is_rank_aware
        self.rank_weight_method = rank_weight_method

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)

    def forward(
        self,
        reading_history: list[str],
        recommendations: list[str],
    ) -> float:
        P_dist = ProbabilityMassFunction.from_list(reading_history)

        Q_dist = (
            RankAwareProbabilityMassFunction.from_ranking(recommendations, self.rank_weight_method)
            if self.is_rank_aware
            else ProbabilityMassFunction.from_list(recommendations)
        )
        calculator = JSDivergence()
        return calculator.calc(P_dist.pmf, Q_dist.pmf)
