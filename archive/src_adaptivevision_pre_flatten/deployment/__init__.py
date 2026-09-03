"""Deployment-profile loading and Pareto recommendation (Milestone M19)."""

from adaptivevision.deployment.profiles import (
    DeploymentProfile,
    explain_recommendation,
    feasible_profiles,
    load_deployment_profiles,
    pareto_frontier,
    recommend,
)

__all__ = [
    "DeploymentProfile",
    "explain_recommendation",
    "feasible_profiles",
    "load_deployment_profiles",
    "pareto_frontier",
    "recommend",
]
