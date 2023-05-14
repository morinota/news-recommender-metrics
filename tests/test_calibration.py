from news_recommender_metrics.RADio.calibration import Calibration


def test_calibration() -> None:
    reading_history = ["b", "a", "a"]
    recommendations = ["a", "b", "b"]

    calibration_expected = Calibration(
        value=0.011127290051423534,  # JS Divergence of P_dist & Q_dist
        P_dist={"a": 0.666, "b": 0.333},
        Q_dist={"a": 0.5454545454545455, "b": 0.45454545454545453},
    )

    calbration_actual = Calibration.calc(
        reading_history,
        recommendations,
        True,
        "MMR",
    )

    assert calbration_actual == calibration_expected
