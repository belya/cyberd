import humanize
from networkx import DiGraph, nx
from scipy.sparse import csr_matrix


def build_matrix(graph: DiGraph, nodes: [str]) -> csr_matrix:
    estimated_size_of_A = humanize.naturalsize(
        graph.number_of_edges() * 8 + graph.number_of_edges() * 4 + graph.number_of_nodes() * 4
    )
    print("Estimated size of A is {} RAM".format(estimated_size_of_A))
    A = nx.to_scipy_sparse_matrix(graph, nodelist=nodes, dtype=float)
    size_of_A = humanize.naturalsize(A.data.nbytes + A.indptr.nbytes + A.indices.nbytes)
    density = A.nnz / (A.shape[0] * A.shape[0])
    print("Matrix A takes {} RAM".format(size_of_A))
    print("Matrix has {:.2e} density".format(density))
    return A
