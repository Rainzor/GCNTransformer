import torch
from torch_geometric.data import Data
from models.data_utils.subgraph_extractors import metis_subgraph, random_subgraph
from models.data_utils.pe import RWSE, LapPE, random_walk
import re


def cal_coarsen_adj(subgraphs_nodes_mask):
    mask = subgraphs_nodes_mask.to(torch.float)
    coarsen_adj = torch.matmul(mask, mask.t())
    return coarsen_adj


def to_sparse(node_mask, edge_mask):
    subgraphs_nodes = node_mask.nonzero().T
    subgraphs_edges = edge_mask.nonzero().T
    return subgraphs_nodes, subgraphs_edges

def combine_subgraphs(edge_index, subgraphs_nodes, subgraphs_edges, num_selected, num_nodes):
    '''
        Combine the subgraphs to a combined subgraph with new node index

        Return:
            [2, all_subgraph_edges]: new edge_index in the combined subgraph
    '''
    # if num_selected is None: # patch_num
    #     num_selected = subgraphs_nodes[0][-1] + 1
    # if num_nodes is None: # node_num
    #     num_nodes = subgraphs_nodes[1].max() + 1

    combined_subgraphs = edge_index[:, subgraphs_edges[1]] # shape: [2, all_subgraph_edges]
    node_label_mapper = edge_index.new_full((num_selected, num_nodes), -1)
    node_label_mapper[subgraphs_nodes[0], subgraphs_nodes[1]
                      ] = torch.arange(len(subgraphs_nodes[1]))
    # map the node index in the subgraph to the index in the combined subgraph
    # e.g  in original graph, there are 5 nodes: 0,1,2,3,4
    # after partition: subgraph 0 contains nodes: 1,2,4
    #                  subgraph 1 contains nodes: 0,2,3
    # then in the combined subgraphs have 6 nodes: 0,1,2,3,4,5
    # the node_label_mapper will be:
    # [[-1, 0, 1, -1, 2],[3, -1, 4, 5, -1]]
    node_label_mapper = node_label_mapper.reshape(-1) # shape: patch_num * node_num

    inc = torch.arange(num_selected)*num_nodes # shape: patch_num
    combined_subgraphs += inc[subgraphs_edges[0]] # to add the offset of the subgraph
    combined_subgraphs = node_label_mapper[combined_subgraphs] 
    return combined_subgraphs # shape: [2, all_subgraph_edges] with new node index in the combined subgraph


class SubgraphsData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        num_nodes = self.num_nodes
        num_edges = self.edge_index.size(-1)
        if bool(re.search('(combined_subgraphs)', key)):
            return getattr(self, key[:-len('combined_subgraphs')]+'subgraphs_nodes_mapper').size(0)
        elif bool(re.search('(subgraphs_batch)', key)):
            return getattr(self, 'mask').size(1)
        elif bool(re.search('(nodes_mapper)', key)):
            return num_nodes
        elif bool(re.search('(edges_mapper)', key)):
            return num_edges
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if bool(re.search('(combined_subgraphs)', key)):
            return -1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


class PositionalEncodingTransform(object):
    def __init__(self, rw_dim=0, lap_dim=0):
        super().__init__()
        self.rw_dim = rw_dim
        self.lap_dim = lap_dim

    def __call__(self, data):
        if self.rw_dim > 0:
            data.rw_pos_enc = RWSE(
                data.edge_index, self.rw_dim, data.num_nodes)
        if self.lap_dim > 0:
            data.lap_pos_enc = LapPE(
                data.edge_index, self.lap_dim, data.num_nodes)
        return data


class GraphPartitionTransform(object):
    def __init__(self, n_patches, metis=True, drop_rate=0.0, num_hops=1, is_directed=False, patch_rw_dim=0, patch_num_diff=0):
        super().__init__()
        self.n_patches = n_patches
        self.drop_rate = drop_rate
        self.num_hops = num_hops
        self.is_directed = is_directed
        self.patch_rw_dim = patch_rw_dim
        self.patch_num_diff = patch_num_diff
        self.metis = metis

    def _diffuse(self, A):
        if self.patch_num_diff == 0:
            return A
        Dinv = A.sum(dim=-1).clamp(min=1).pow(-1).unsqueeze(-1)  # D^-1
        RW = A * Dinv
        M = RW
        M_power = M
        # Iterate
        for _ in range(self.patch_num_diff-1):
            M_power = torch.matmul(M_power, M)
        return M_power

    def __call__(self, data):
        data = SubgraphsData(**{k: v for k, v in data})
        if self.metis:
            node_masks, edge_masks = metis_subgraph(
                data, n_patches=self.n_patches, drop_rate=self.drop_rate, num_hops=self.num_hops, is_directed=self.is_directed)
        else:
            node_masks, edge_masks = random_subgraph(
                data, n_patches=self.n_patches, num_hops=self.num_hops)

        # data.patch_pos = torch.stack([pos[node_masks[i]].mean(0)
        #                   for i in range(node_masks.size(0))])

        data.patch_pos = torch.zeros(self.n_patches, data.pos.size(1))
        for i in range(node_masks.size(0)):
            if node_masks[i].sum() == 0:
                continue
            data.patch_pos[i] = data.pos[node_masks[i]].mean(0)

        subgraphs_nodes, subgraphs_edges = to_sparse(node_masks, edge_masks)
        combined_subgraphs = combine_subgraphs(
            data.edge_index, subgraphs_nodes, subgraphs_edges, num_selected=self.n_patches, num_nodes=data.num_nodes)

        if self.patch_num_diff > -1 or self.patch_rw_dim > 0:
            coarsen_adj = cal_coarsen_adj(node_masks)
            if self.patch_rw_dim > 0:
                data.patch_pe = random_walk(coarsen_adj, self.patch_rw_dim)
            if self.patch_num_diff > -1:
                data.coarsen_adj = self._diffuse(coarsen_adj).unsqueeze(0)

        subgraphs_batch = subgraphs_nodes[0] # shape: [all_nodes_in_combined_subgraph]
        mask = torch.zeros(self.n_patches).bool()
        mask[subgraphs_batch] = True 
        data.subgraphs_batch = subgraphs_batch 
        # value is original node index, idx is new node index in the combined subgraph
        data.subgraphs_nodes_mapper = subgraphs_nodes[1] # shape: [all_nodes_in_combined_subgraph]
        # value is original edge index, idx is new edge index in the combined subgraph 
        data.subgraphs_edges_mapper = subgraphs_edges[1] # shape: [all_edges_in_combined_subgraph]
        data.combined_subgraphs = combined_subgraphs # shape: [2, all_subgraph_edges] with new node index in the combined subgraph
        data.mask = mask.unsqueeze(0) # shape: [1, n_patches] to identify the subgraph has nodes or not

        data.__num_nodes__ = data.num_nodes  # set number of nodes of the current graph
        return data
