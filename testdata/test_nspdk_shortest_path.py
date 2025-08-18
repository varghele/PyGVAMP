import torch
import torch_sparse                        # pip install torch-sparse
import time

def all_pairs_up_to_d(edge_index, num_nodes, d_max):
    # edge_index shape (2,E), undirected
    # ---------------------------------------------------------------------
    A = torch.sparse_coo_tensor(edge_index,      # indices
                                torch.ones(edge_index.size(1)),
                                (num_nodes, num_nodes),
                                dtype=torch.bool).coalesce()

    # visited[i,j] == shortest distance found so far (0-based, -1 == unseen)
    visited = -torch.ones(num_nodes, num_nodes, dtype=torch.int16,
                          device=A.device)

    # distance 0   (diagonal)
    idx = torch.arange(num_nodes, device=A.device)
    visited[idx, idx] = 0

    frontier = A                    # all pairs at distance 1
    k = 1
    row_all, col_all, dist_all = [], [], []

    while k <= d_max and frontier._nnz() > 0:
        r, c = frontier.indices()

        # keep pairs never seen before
        mask = visited[r, c] < 0
        r, c = r[mask], c[mask]
        if r.numel():                      # anything new?
            visited[r, c] = k              # store shortest distance
            row_all.append(r)
            col_all.append(c)
            dist_all.append(torch.full_like(r, k, dtype=torch.int16))

        # next frontier :   A^{k+1} = A^k @ A
        # Fix: convert result back to COO format before coalesce
        indices, values = torch_sparse.spspmm(
            frontier.indices(), frontier.values().float(),
            A.indices(), A.values().float(),
            num_nodes, num_nodes, num_nodes)

        frontier = torch.sparse_coo_tensor(
            indices, values, (num_nodes, num_nodes), dtype=torch.bool
        ).coalesce()

        k += 1

    # build sparse distance matrix (only pairs within d_max)
    if row_all:
        rows  = torch.cat(row_all)
        cols  = torch.cat(col_all)
        dists = torch.cat(dist_all)
        dist_sp = torch.sparse_coo_tensor(
            torch.stack([rows, cols]), dists,
            (num_nodes, num_nodes), dtype=torch.int16).coalesce()
    else:                                 # no pair within d_max
        dist_sp = torch.sparse_coo_tensor(
            torch.empty(2,0, dtype=torch.long, device=A.device),
            torch.empty(0, dtype=torch.int16, device=A.device),
            (num_nodes, num_nodes), dtype=torch.int16)

    return dist_sp        # (row, col, value=k) for every shortest-path ≤ d_max


# Create a simple graph: 0-1-2-3 (linear chain)
#edge_index = torch.tensor([
#    [0, 1, 1, 2, 2, 3],  # source nodes
#    [1, 0, 2, 1, 3, 2]   # target nodes
#], dtype=torch.long)

#num_nodes = 4
d_max = 5

#print("Graph structure:")
#print("Nodes: 0, 1, 2, 3")
#print("Edges: 0-1, 1-2, 2-3 (undirected)")
#print()

# ---------- parameters ----------
num_nodes = 1100          # 10k vertices
device     = 'cpu'          # or 'cuda' if you have enough GPU RAM
dtype      = torch.long
# --------------------------------

# Upper-triangular indices (i < j)  –  gives each edge only once
row, col = torch.triu_indices(
    num_nodes, num_nodes, offset=1, device=device, dtype=dtype)

# row, col together hold 10k·(10k-1)/2 = 49 995 000 edges
# To make the graph undirected we add the reversed edges:
edge_index = torch.cat([                       # shape = (2, 99 990 000)
    torch.stack([row, col], dim=0),            #  i -> j
    torch.stack([col, row], dim=0)             #  j -> i
], dim=1)

print(edge_index.shape)        # (2, 99_990_000)
print(edge_index.element_size() * edge_index.numel() / 1e9, "GB")


start = time.perf_counter()
# Run the function
dist_sp = all_pairs_up_to_d(edge_index, num_nodes, d_max)

torch.cuda.synchronize()        # if running on GPU
elapsed = time.perf_counter() - start

print(f"Finished in {elapsed:.2f} s")
print(dist_sp._nnz(), "pairs within distance ≤", d_max)


print("Results:")
print("Sparse distance matrix indices:", dist_sp.indices())
print("Distance values:", dist_sp.values())
print()

# Convert to readable format
rows, cols = dist_sp.indices()
dists = dist_sp.values()

print("All pairs with distance ≤", d_max, ":")
#for i in range(len(rows)):
#    print(f"Node {rows[i].item()} to Node {cols[i].item()}: distance {dists[i].item()}")
