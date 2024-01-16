import math

from news_recommender_metrics.RADio.calibration import Calibration


def test_calibration() -> None:
    # Arrange
    reading_history_of_user_A = ["bussiness", "technology", "technology"]
    recommended_topics_of_user_A = ["technology", "bussiness", "bussiness"]
    sut = Calibration(
        is_rank_aware=True,
        rank_weight_method="MMR",
    )

    # Act
    calbration_actual = sut(reading_history_of_user_A, recommended_topics_of_user_A)

    # Assert
    value_expected = 0.011127  # JS Divergence of P_dist & Q_dist
    assert math.isclose(calbration_actual, value_expected, rel_tol=1e-2)
