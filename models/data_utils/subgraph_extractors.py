import torch
from torch_sparse import SparseTensor  # for propagation
import numpy as np
import metis
import pymetis
import torch_geometric
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

def k_hop_subgraph(edge_index, num_nodes, num_hops, is_directed=False):
    # return k-hop subgraphs for all nodes in the graph
    if is_directed:
        row, col = edge_index
        birow, bicol = torch.cat([row, col]), torch.cat([col, row])
        edge_index = torch.stack([birow, bicol])
    else:
        row, col = edge_index
    sparse_adj = SparseTensor(
        row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
    # each one contains <= i hop masks
    hop_masks = [torch.eye(num_nodes, dtype=torch.bool,
                           device=edge_index.device)]
    hop_indicator = row.new_full((num_nodes, num_nodes), -1)
    hop_indicator[hop_masks[0]] = 0
    for i in range(num_hops):
        next_mask = sparse_adj.matmul(hop_masks[i].float()) > 0
        hop_masks.append(next_mask)
        hop_indicator[(hop_indicator == -1) & next_mask] = i+1
    hop_indicator = hop_indicator.T  # N x N
    node_mask = (hop_indicator >= 0)  # N x N dense mask matrix
    return node_mask


def random_subgraph(g, n_patches, num_hops=1):
    membership = np.arange(g.num_nodes)
    np.random.shuffle(membership)
    membership = torch.tensor(membership % n_patches)
    max_patch_id = torch.max(membership)+1
    membership = membership+(n_patches-max_patch_id)

    node_mask = torch.stack([membership == i for i in range(n_patches)])

    if num_hops > 0:
        subgraphs_batch, subgraphs_node_mapper = node_mask.nonzero().T
        k_hop_node_mask = k_hop_subgraph(
            g.edge_index, g.num_nodes, num_hops)
        node_mask[subgraphs_batch] += k_hop_node_mask[subgraphs_node_mapper]

    edge_mask = node_mask[:, g.edge_index[0]] & node_mask[:, g.edge_index[1]]
    return node_mask, edge_mask

def sort_subgraphs(pos, node_mask):
    """Sort the subgraphs based on the node positions.

    Parameters
    ----------
    pos : torch.Tensor [num_nodes, 2]
        The node positions.
    node_mask : torch.Tensor [n_patches, num_nodes]
        The node mask for each subgraph.

    Returns
    -------
    node_mask : torch.Tensor [n_patches, num_nodes]
        The node mask for each subgraph.
    """
    # calculate the center of each subgraph
    mask = torch.sum(node_mask, dim=1) > 0
    compressed_mask = node_mask[mask]
    compressed_len = compressed_mask.size(0)
    center = torch.stack([pos[compressed_mask[i]].mean(0)
                          for i in range(compressed_len)])
    radius = torch.norm(center, dim=1)
    # sort the subgraphs based on the radius
    _, indices = torch.sort(radius)
    sorted_node_mask = node_mask.new_full(node_mask.size(), False)
    sorted_node_mask[:compressed_len,:] = compressed_mask[indices]
    return sorted_node_mask

def metis_subgraph(g, n_patches, drop_rate=0.0, num_hops=1, is_directed=False):
    """Partition the graph into subgraphs using METIS or random partitioning.

    Parameters
    ----------
    g : pytorch_geometric.data.Data
        The input graph.
    n_patches : int
        The number of subgraphs.
    drop_rate : float, optional
        The drop rate for edges to augment the graph, by default 0.0
    num_hops : int, optional
        The overlap of subgraphs, by default 1
    is_directed : bool, optional
        Whether the graph is directed, by default False

    Returns
    -------
    node_mask : torch.Tensor [n_patches, num_nodes]
        The node mask for each subgraph.
    edge_mask : torch.Tensor [n_patches, num_edges]
        The edge mask for each subgraph
    """
    if is_directed:
        if g.num_nodes < n_patches:
            membership = torch.arange(g.num_nodes)
        else:
            G = torch_geometric.utils.to_networkx(g, to_undirected="lower")
            cuts, membership = metis.part_graph(G, n_patches, recursive=True)
    else:
        if g.num_nodes < n_patches:
            membership = torch.randperm(n_patches)
        else:
            # data augmentation
            adjlist = g.edge_index.t()
            arr = torch.rand(len(adjlist))
            selected = arr > drop_rate
            G = nx.Graph()
            G.add_nodes_from(np.arange(g.num_nodes))
            G.add_edges_from(adjlist[selected].tolist())
            # metis partition
            cuts, membership = metis.part_graph(G, n_patches, recursive=True)

    # membership is a list of subgraph partition ids for each node
    assert len(membership) >= g.num_nodes
    membership = torch.tensor(np.array(membership[:g.num_nodes])) # Shape: [num_nodes]
    max_patch_id = torch.max(membership)+1
    membership = membership+(n_patches-max_patch_id)

    # node_mask is a list of boolean masks for each subgraph
    node_mask = torch.stack([membership == i for i in range(n_patches)]) # Shape: [n_patches, num_nodes]
    if (getattr(g, 'pos', None) is not None):
        node_mask = sort_subgraphs(g.pos, node_mask)
    if num_hops > 0:
        # subgraphs_batch is the batch id for each subgraph
        # subgraphs_node_mapper is the node id in the original graph for each subgraph node
        # e.g. sb[0] and snm[0] means the node id snm[0] in the original graph is in the subgraph sb[0]
        subgraphs_batch, subgraphs_node_mapper = node_mask.nonzero().T

        # k_hop_node_mask is the k-hop mask for each node in the original graph
        # especially, diagonal elements are set to True
        k_hop_node_mask = k_hop_subgraph(
            g.edge_index, g.num_nodes, num_hops, is_directed)
        node_mask.index_add_(0, subgraphs_batch,
                             k_hop_node_mask[subgraphs_node_mapper])

    # restrict that the edge's two nodes must be in the same subgraph,
    # then the edge is in the subgraph
    edge_mask = node_mask[:, g.edge_index[0]] & node_mask[:, g.edge_index[1]] # Shape: [n_patches, num_edges]
    return node_mask, edge_mask

# Define the metis_subgraph function
def pymetis_subgraph(g, n_patches, drop_rate=0.0, num_hops=1, is_directed=False):
    if is_directed:
        if g.num_nodes < n_patches:
            membership = torch.arange(g.num_nodes)
        else:
            # Convert directed graph to undirected graph
            G = torch_geometric.utils.to_networkx(g, to_undirected="lower")
            # Use pymetis for graph partitioning
            adjacency = [list(G.adj[i]) for i in range(g.num_nodes)]
            num_cuts, membership = pymetis.part_graph(n_patches, adjacency=adjacency)
    else:
        if g.num_nodes < n_patches:
            # If the number of nodes is less than the number of patches, assign randomly
            membership = torch.randperm(n_patches)[:g.num_nodes]
        else:
            # Data augmentation: randomly remove some edges
            adjlist = g.edge_index.t().numpy()
            arr = np.random.rand(len(adjlist))
            selected = arr > drop_rate
            G = nx.Graph()
            G.add_nodes_from(np.arange(g.num_nodes))
            G.add_edges_from(adjlist[selected].tolist())
            # Use pymetis for graph partitioning
            adjacency = [list(G.adj[i]) for i in range(g.num_nodes)]
            num_cuts, membership = pymetis.part_graph(n_patches, adjacency=adjacency)

    # Ensure that the length of membership is not less than the number of nodes
    assert len(membership) >= g.num_nodes
    membership = torch.tensor(np.array(membership[:g.num_nodes]))  # Shape: [num_nodes]
    max_patch_id = torch.max(membership) + 1
    membership = membership + (n_patches - max_patch_id)

    # Create boolean masks for each subgraph
    node_mask = torch.stack([membership == i for i in range(n_patches)])  # Shape: [n_patches, num_nodes]

    if num_hops > 0:
        # Get node indices for each subgraph
        subgraphs_batch, subgraphs_node_mapper = node_mask.nonzero(as_tuple=True)
        # Compute k-hop subgraph mask
        k_hop_node_mask = k_hop_subgraph(
            g.edge_index, g.num_nodes, num_hops, is_directed)
        # Update node_mask to include k-hop neighbors
        node_mask.index_add_(0, subgraphs_batch, k_hop_node_mask[subgraphs_node_mapper])

        # Convert node_mask to boolean type
        node_mask = node_mask.bool()

    # Create edge masks for each subgraph
    edge_mask = node_mask[:, g.edge_index[0]] & node_mask[:, g.edge_index[1]]
    return node_mask, edge_mask

def kmeans_subgraph(g, n_patches, drop_rate=0.0, num_hops=1, is_directed=False):
    """
    Use KMeans clustering to partition the graph into subgraphs.

    Parameters:
        g (torch_geometric.data.Data):
        n_patches (int): expected number of subgraphs.
        drop_rate (float): data augmentation parameter to randomly remove edges.
        num_hops (int): k-hop neighbors to include in each subgraph.
        is_directed (bool): whether the graph is directed.

    Returns:
        node_mask (torch.BoolTensor): every subgraph's node mask, shape [n_patches, num_nodes].
        edge_mask (torch.BoolTensor): every subgraph's edge mask, shape [n_patches, num_edges].
    """
    num_nodes = g.num_nodes

    if num_nodes < n_patches:
        # 如果节点数少于簇数，随机分配簇
        membership = torch.randperm(n_patches)[:num_nodes]
    else:
        # 提取节点的位置信息作为聚类特征
        pos = g.x.numpy()  # 假设 g.x 包含节点的位置信息

        # 计算点云的边界
        x_min, y_min = pos.min(axis=0)
        x_max, y_max = pos.max(axis=0)

        # 生成均匀分布的初始中心点
        # 假设 n_patches 是一个完全平方数，以便在网格中均匀分布
        grid_size = int(np.ceil(np.sqrt(n_patches)))
        x_lin = np.linspace(x_min, x_max, grid_size)
        y_lin = np.linspace(y_min, y_max, grid_size)
        xv, yv = np.meshgrid(x_lin, y_lin)
        initial_centers = np.vstack([xv.ravel(), yv.ravel()]).T[:n_patches]

        # 使用均匀初始化的中心点进行 KMeans 聚类
        kmeans = KMeans(n_clusters=n_patches, init=initial_centers, n_init=1, random_state=42)
        membership = kmeans.fit_predict(pos)
        membership = torch.tensor(membership, dtype=torch.long)
    
    # 创建每个簇的掩码
    node_mask = torch.stack([membership == i for i in range(n_patches)])  # 形状: [n_patches, num_nodes]

    if num_hops > 0:
        # 获取子图的节点索引
        subgraphs_batch, subgraphs_node_mapper = node_mask.nonzero(as_tuple=True)
        # 计算 k-hop 子图掩码
        k_hop_node_mask = k_hop_subgraph(
            g.edge_index, g.num_nodes, num_hops, is_directed)
        # 更新 node_mask 以包含 k-hop 邻居
        node_mask.index_add_(0, subgraphs_batch, k_hop_node_mask[subgraphs_node_mapper])

        # 将 node_mask 转换为布尔类型
        node_mask = node_mask.bool()

    # 创建每个子图的边掩码
    edge_mask = node_mask[:, g.edge_index[0]] & node_mask[:, g.edge_index[1]]
    return node_mask, edge_mask
