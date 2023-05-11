import math

from news_recommender_metrics.RADio.divergence_metrics import JSDivergence, KLDivergence


def test_calc_KL_divergence() -> None:
    P = {"a": 0.2, "b": 0.3, "c": 0.5}
    Q = {"a": 0.3, "b": 0.3, "c": 0.4}
    kl_div_PQ_expected = -(0.2 * math.log2(0.3) + 0.3 * math.log2(0.3) + 0.5 * math.log2(0.4)) + (
        0.2 * math.log2(0.2) + 0.3 * math.log2(0.3) + 0.5 * math.log2(0.5)
    )

    calculator = KLDivergence()
    kl_div_PQ_actual = calculator.calc(P, Q)
    assert math.isclose(
        kl_div_PQ_actual,
        kl_div_PQ_expected,
        rel_tol=1e-3,
    )  # The value varies slightly depending on the adjustment (JSDivergence._convert_to_valid_dist) to make the probability distribution valid.


def test_calc_JS_divergence() -> None:
    P = {"a": 0.2, "b": 0.3, "c": 0.5}
    Q = {"a": 0.3, "b": 0.3, "c": 0.4}

    js_div_first_term = (
        -(0.2 + 0.3) / 2 * math.log2((0.2 + 0.3) / 2)
        - (0.3 + 0.3) / 2 * math.log2((0.3 + 0.3) / 2)
        - (0.5 + 0.4) / 2 * math.log2((0.5 + 0.4) / 2)
    )
    js_div_second_term = 1 / 2 * (0.2 * math.log2(0.2) + 0.3 * math.log2(0.3) + 0.5 * math.log2(0.5))
    js_div_third_term = 1 / 2 * (0.3 * math.log2(0.3) + 0.3 * math.log2(0.3) + 0.4 * math.log2(0.4))
    js_div_PQ_expected = js_div_first_term + js_div_second_term + js_div_third_term

    calculator = JSDivergence()
    js_div_PQ_actual = calculator.calc(P, Q)
    assert math.isclose(
        js_div_PQ_actual,
        js_div_PQ_expected,
        rel_tol=1e-3,
    )  # The value varies slightly depending on the adjustment (JSDivergence._convert_to_valid_dist) to make the probability distribution valid.
