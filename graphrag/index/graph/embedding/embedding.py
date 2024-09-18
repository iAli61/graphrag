import logging
from dataclasses import dataclass
import graspologic as gc
import networkx as nx
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NodeEmbeddings:
    """Node embeddings class definition."""
    nodes: list[str]
    embeddings: np.ndarray

def embed_nod2vec(
    graph: nx.Graph | nx.DiGraph,
    dimensions: int = 1536,
    num_walks: int = 10,
    walk_length: int = 40,
    window_size: int = 2,
    iterations: int = 3,
    random_seed: int = 86,
) -> NodeEmbeddings:
    """Generate node embeddings using Node2Vec."""
    # Check if the graph is empty
    if graph.number_of_nodes() == 0:
        logger.warning("The input graph is empty. Returning empty embeddings.")
        return NodeEmbeddings(nodes=[], embeddings=np.array([]))

    # Check for isolated nodes
    isolated_nodes = list(nx.isolates(graph))
    if isolated_nodes:
        logger.warning(f"Graph contains {len(isolated_nodes)} isolated nodes. These will be removed.")
        graph = graph.copy()
        graph.remove_nodes_from(isolated_nodes)

    # Check if the graph is still non-empty after removing isolated nodes
    if graph.number_of_nodes() == 0:
        logger.warning("All nodes were isolated. Returning empty embeddings.")
        return NodeEmbeddings(nodes=[], embeddings=np.array([]))

    try:
        # generate embedding
        lcc_tensors = gc.embed.node2vec_embed(
            graph=graph,
            dimensions=dimensions,
            window_size=window_size,
            iterations=iterations,
            num_walks=num_walks,
            walk_length=walk_length,
            random_seed=random_seed,
        )
        return NodeEmbeddings(embeddings=lcc_tensors[0], nodes=lcc_tensors[1])
    except ZeroDivisionError:
        logger.error("ZeroDivisionError occurred during Node2Vec embedding. This might be due to nodes with no valid transitions.")
        return NodeEmbeddings(nodes=list(graph.nodes), embeddings=np.zeros((graph.number_of_nodes(), dimensions)))
    except Exception as e:
        logger.error(f"An unexpected error occurred during Node2Vec embedding: {str(e)}")
        raise