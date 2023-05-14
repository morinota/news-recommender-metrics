from typing import Any, List

from news_recommender_metrics.RADio.dart_metrics_abstract import DartMetricsAbstract
from news_recommender_metrics.RADio.rank_aware_probability_mass_function import RankAwareProbabilityMassFunction


class Calibration(DartMetricsAbstract):
    @classmethod
    def calc(
        cls,
        reading_history: List[Any],
        recommendations: List[Any],
        is_rank_aware: bool = True,
        rank_weight_method: str = "MMR",
    ) -> "Calibration":
        """_summary_

        Parameters
        ----------
        reading_history : List[Any]
            the user's reading history of news. element is an item attribute for calculate Calibration.(ex. news category label)
        recommendations : List[Any]
            recommendation result for the user. it's assumed being sorted by recommendation ranking.
        is_rank_aware : bool
            if True, calculate Calibration with rank-aware distribution of recommendations.
            Else, calculate without considering ranking.
        rank_weight_method: str = "MMR"
            the method for weighting of the rank.("MMR" or "nDCG")
        Returns
        -------
        Calibration
            Calibration of the item attribute.
        """
        P = RankAwareProbabilityMassFunction.from_ranking(reading_history)
        Q = RankAwareProbabilityMassFunction.from_ranking(recommendations, rank_weight_method)
