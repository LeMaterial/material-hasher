# Copyright 2025 Entalpic
from networkx import Graph
from structuregraph_helpers.hash import generate_hash


def get_weisfeiler_lehman_hash(
    graph: Graph,
) -> str:
    """Builds Weisfeiler Lehman hash from Graph object

    Args:
        graph (Graph): Graph object

    Returns:
        str: Weisfeiler Lehman hash
    """
    return generate_hash(graph, True, False, 100)
