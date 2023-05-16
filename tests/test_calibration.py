import math

from news_recommender_metrics.RADio.calibration import Calibration


def test_calibration() -> None:
    reading_history = ["b", "a", "a"]
    recommendations = ["a", "b", "b"]

    calbration_actual = Calibration.calc(
        reading_history,
        recommendations,
        True,
        "MMR",
    )

    P_dist_expected = {"a": 0.666, "b": 0.333}
    Q_dist_expected = {
        "a": 0.545454545,
        "b": 0.454545454,
    }
    value_expected = 0.011127  # JS Divergence of P_dist & Q_dist
    assert [
        math.isclose(prob_mass, P_dist_expected.get(x, 0))
        for x, prob_mass in calbration_actual.P_dist.items()
    ]
    assert [
        math.isclose(prob_mass, Q_dist_expected.get(x, 0))
        for x, prob_mass in calbration_actual.Q_dist.items()
    ]
    assert math.isclose(round(calbration_actual.value, 6), value_expected)
