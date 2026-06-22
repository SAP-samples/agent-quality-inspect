from typing import List

from agent_inspect.models.tools.error_cluster import ErrorCluster
from agent_inspect.exception import InvalidInputValueError
from agent_inspect.exception.error_codes import ToolComponent, ErrorCode


def validate_user_defined_clusters(clusters: List[ErrorCluster]) -> None:
    """Validate that clusters are properly formatted ErrorCluster instances."""
    if not isinstance(clusters, list):
        raise InvalidInputValueError(
            internal_code=ErrorCode.INVALID_ERROR_ANALYSIS_ERROR_CLUSTERS.value,
            message=f"Clusters must be a list of ErrorCluster instances. Got: {type(clusters).__name__}",
            component_code=ToolComponent.TOOL_ERROR_CODE.value,
        )

    for idx, cluster in enumerate(clusters):
        if not isinstance(cluster, ErrorCluster):
            raise InvalidInputValueError(
                internal_code=ErrorCode.INVALID_ERROR_ANALYSIS_ERROR_CLUSTERS.value,
                message=f"Cluster at index {idx} must be a ErrorCluster instance. Got: {type(cluster).__name__}",
                component_code=ToolComponent.TOOL_ERROR_CODE.value,
            )

        # Check for empty cluster label
        if not cluster.cluster_label or not isinstance(cluster.cluster_label, str):
            raise InvalidInputValueError(
                internal_code=ErrorCode.INVALID_ERROR_ANALYSIS_ERROR_CLUSTERS.value,
                message=f"Cluster at index {idx} has invalid 'cluster_label'. Must be a non-empty string. Got: {cluster.cluster_label}",
                component_code=ToolComponent.TOOL_ERROR_CODE.value,
            )

        # Check for empty description
        if not cluster.description or not isinstance(cluster.description, str):
            raise InvalidInputValueError(
                internal_code=ErrorCode.INVALID_ERROR_ANALYSIS_ERROR_CLUSTERS.value,
                message=f"Cluster at index {idx} has invalid 'description'. Must be a non-empty string. Got: {cluster.description}",
                component_code=ToolComponent.TOOL_ERROR_CODE.value,
            )
