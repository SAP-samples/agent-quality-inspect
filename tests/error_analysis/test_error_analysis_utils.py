from dataclasses import dataclass
import pytest

from agent_inspect.exception.exception import InvalidInputValueError
from agent_inspect.models.tools.error_cluster import (
    ErrorCluster,
    IncorrectToolInputCluster,
    IncorrectToolOutputHandlingCluster,
    WrongToolSelectionCluster,
)
from agent_inspect.tools.error_analysis.utils.validator import (
    validate_user_defined_clusters,
)


def test_validate_clusters_with_valid_default_clusters():
    clusters = [
        IncorrectToolInputCluster(),
        WrongToolSelectionCluster(),
        IncorrectToolOutputHandlingCluster(),
    ]
    validate_user_defined_clusters(clusters)


def test_validate_clusters_with_single_valid_cluster():
    validate_user_defined_clusters([IncorrectToolInputCluster()])


def test_validate_clusters_with_valid_custom_clusters():

    @dataclass(frozen=True)
    class CustomCluster(ErrorCluster):
        cluster_label: str = "Custom Cluster"
        description: str = "A custom error cluster"

    clusters = [CustomCluster()]
    validate_user_defined_clusters(clusters)


def test_validate_clusters_raises_for_empty_label():
    @dataclass(frozen=True)
    class EmptyLabelCluster(ErrorCluster):
        cluster_label: str = ""
        description: str = "Some description"

    with pytest.raises(InvalidInputValueError, match="invalid 'cluster_label'"):
        validate_user_defined_clusters([EmptyLabelCluster()])


def test_validate_clusters_raises_for_empty_description():
    @dataclass(frozen=True)
    class EmptyDescCluster(ErrorCluster):
        cluster_label: str = "Some Label"
        description: str = ""

    with pytest.raises(InvalidInputValueError, match="invalid 'description'"):
        validate_user_defined_clusters([EmptyDescCluster()])


def test_validate_clusters_raises_for_non_list_input():
    with pytest.raises(InvalidInputValueError, match="must be a list"):
        validate_user_defined_clusters("not a list")


def test_validate_clusters_raises_for_none_input():
    with pytest.raises(InvalidInputValueError, match="must be a list"):
        validate_user_defined_clusters(None)


def test_validate_clusters_raises_for_non_base_error_cluster():
    with pytest.raises(InvalidInputValueError, match="must be a ErrorCluster instance"):
        validate_user_defined_clusters(
            [
                IncorrectToolInputCluster(),
                {"cluster_label": "bad", "description": "not a dataclass"},
            ]
        )
