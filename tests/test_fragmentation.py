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
    assert math.isclose(fragmentation_actual, 0.0, rel_tol=1e-2)


def test_fragmentation_with_different_recommend_results() -> None:
    # Arrange
    recommend_topics_of_user_A = ["bussiness", "sports", "finance"]
    recommend_topics_of_user_B = ["technology", "science", "politics"]
    sut = Fragmentation(
        is_rank_aware=True,
        rank_weight_method="MMR",
    )

    # Act
    fragmentation_actual = sut(recommend_topics_of_user_A, recommend_topics_of_user_B)

    # Assert
    assert not math.isclose(fragmentation_actual, 0.0, rel_tol=1e-2)
    assert math.isclose(fragmentation_actual, 1.0, rel_tol=1e-2)
