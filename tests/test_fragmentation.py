import math

from news_recommender_metrics.RADio.fragmentation import Fragmentation


def test_fragmentation_with_same_recommend_results() -> None:
    # Arrange
    recommend_topics_of_user_A = ["bussiness", "sports", "finance"]
    recommend_topics_of_user_B = ["bussiness", "sports", "finance"]
    sut = Fragmentation(
        is_rank_aware=True,
        rank_weight_method="MMR",
    )

    # Act
    fragmentation_actual = sut(recommend_topics_of_user_A, recommend_topics_of_user_B)

    # Assert
    assert math.isclose(fragmentation_actual, 1.0, rel_tol=1e-3)


def test_fragmentation_with_different_recommend_results() -> None:
    # Arrange
    recommend_topics_of_user_A = ["bussiness", "sports", "finance"]
    recommend_topics_of_user_B = ["finance", "sports", "bussiness"]
    sut = Fragmentation(
        is_rank_aware=True,
        rank_weight_method="MMR",
    )

    # Act
    fragmentation_actual = sut(recommend_topics_of_user_A, recommend_topics_of_user_B)

    # Assert
    fragmentation_expected = 0.5
    assert math.isclose(fragmentation_actual, fragmentation_expected, rel_tol=1e-3)
