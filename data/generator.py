import torch
import random
import numpy as np
import os
import sys
import torch_geometric
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

# Specify your METIS_DLL path
# os.system('export METIS_DLL=~/.local/lib/libmetis.so')
import metispy as metis


def get_data(dataset, data_path):
    if dataset in ['Cora', 'CiteSeer', 'PubMed']:
        data = torch_geometric.datasets.Planetoid(data_path, dataset, transform=T.Compose([LargestConnectedComponents(), T.NormalizeFeatures()]))[0]
    elif dataset in ['Computers', 'Photo']:
        data = torch_geometric.datasets.Amazon(data_path, dataset, transform=T.Compose([LargestConnectedComponents(), T.NormalizeFeatures()]))[0]
        data.train_mask, data.val_mask, data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool)
    elif dataset in ['ogbn-arxiv']:
        data = PygNodePropPredDataset(dataset, root=data_path, transform=T.Compose([T.ToUndirected(), LargestConnectedComponents()]))[0]
        data.train_mask, data.val_mask, data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool)
        data.y = data.y.view(-1)
    return data


def split_train(data, dataset, data_path, ratio_train, mode, n_clients):
    n_data = data.num_nodes
    ratio_test = (1-ratio_train)/2
    n_train = round(n_data * ratio_train)
    n_test = round(n_data * ratio_test)

    permuted_indices = torch.randperm(n_data)
    train_indices, test_indices, val_indices = permuted_indices[:n_train], permuted_indices[n_train:n_train+n_test], permuted_indices[n_train+n_test:]

    data.train_mask.fill_(False), data.test_mask.fill_(False), data.val_mask.fill_(False)
    data.train_mask[train_indices] = data.test_mask[test_indices] = data.val_mask[val_indices] = True

    torch_save(os.path.join(data_path, f'{dataset}_{mode}/{n_clients}'), f'train.pt', {'data': data})
    torch_save(os.path.join(data_path, f'{dataset}_{mode}/{n_clients}'), f'test.pt', {'data': data})
    torch_save(os.path.join(data_path, f'{dataset}_{mode}/{n_clients}'), f'val.pt', {'data': data})
    print(f'splition done, n_train: {n_train}, n_test: {n_test}, n_val: {len(val_indices)}')
    return data


class LargestConnectedComponents(T.BaseTransform):
    r"""Selects the subgraph that corresponds to the
    largest connected components in the graph.

    Args:
        num_components (int, optional): Number of largest components to keep
            (default: :obj:`1`)
    """

    def __init__(self, num_components: int = 1):
        self.num_components = num_components

    def __call__(self, data: torch_geometric.data.Data) -> torch_geometric.data.Data:
        import numpy as np
        import scipy.sparse as sp

        adj = torch_geometric.utils.to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)

        num_components, component = sp.csgraph.connected_components(adj)

        if num_components <= self.num_components:
            return data

        _, count = np.unique(component, return_counts=True)
        subset = np.in1d(component, count.argsort()[-self.num_components:])

        return data.subgraph(torch.from_numpy(subset).to(torch.bool))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num_components})'


def get_client_data(data, client_indices, adj):
    client_adj = adj[client_indices][:, client_indices]
    client_edge_index, _ = torch_geometric.utils.dense_to_sparse(client_adj)
    client_edge_index = client_edge_index.T.tolist()

    client_edge_index = torch.tensor(client_edge_index, dtype=torch.long)
    client_x, client_y = data.x[client_indices], data.y[client_indices]
    client_train_mask, client_val_mask, client_test_mask = data.train_mask[client_indices], data.val_mask[client_indices], data.test_mask[client_indices]

    client_data = torch_geometric.data.Data(
        x=client_x,
        y=client_y,
        edge_index=client_edge_index.t().contiguous(),
        train_mask=client_train_mask,
        val_mask=client_val_mask,
        test_mask=client_test_mask
    )
    assert torch.sum(client_train_mask).item() > 0

    return client_data, len(client_indices), len(client_edge_index)


def torch_save(base_dir, filename, data):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    fpath = os.path.join(base_dir, filename)
    torch.save(data, fpath)


def split_disjoint_subgraphs(n_clients, data, dataset, data_path):
    G = torch_geometric.utils.to_networkx(data)
    n_cuts, membership = metis.part_graph(G, n_clients)
    assert len(list(set(membership))) == n_clients
    print(f'graph partition done, metis, n_partitions: {len(list(set(membership)))}, n_lost_edges: {n_cuts}')

    adj = torch_geometric.utils.to_dense_adj(data.edge_index)[0]
    for client_id in range(n_clients):

        client_indices = list(np.where(np.array(membership) == client_id)[0])
        client_data, client_num_nodes, client_num_edges = get_client_data(data, client_indices, adj)

        torch_save(os.path.join(data_path, f'{dataset}_disjoint/{n_clients}'), f'partition_{client_id}.pt', {
            'client_data': client_data,
            'client_id': client_id
        })
        print(f'client_id: {client_id}, iid, n_train_node: {client_num_nodes}, n_train_edge: {client_num_edges}')


def split_overlapping_subgraphs(n_comms, n_client_per_comm, data, dataset, data_path):
    G = torch_geometric.utils.to_networkx(data)
    n_cuts, membership = metis.part_graph(G, n_comms)
    assert len(list(set(membership))) == n_comms
    print(f'graph partition done, metis, n_partitions: {len(list(set(membership)))}, n_lost_edges: {n_cuts}')

    adj = torch_geometric.utils.to_dense_adj(data.edge_index)[0]
    for comm_id in range(n_comms):
        for client_id in range(n_client_per_comm):

            client_indices = list(np.where(np.array(membership) == comm_id)[0])
            client_indices = random.sample(client_indices, len(client_indices) // 2)
            client_data, client_num_nodes, client_num_edges = get_client_data(data, client_indices, adj)

            torch_save(data_path, f'{dataset}_overlapping/{n_comms*n_client_per_comm}/partition_{comm_id*n_client_per_comm+client_id}.pt', {
                'client_data': client_data,
                'client_id': client_id
            })
            print(f'client_id: {comm_id*n_client_per_comm+client_id}, iid, n_train_node: {client_num_nodes}, n_train_edge: {client_num_edges}')


def generate_disjoint_data(dataset, n_clients, data_path, ratio_train):
    data = split_train(get_data(dataset, data_path), dataset, data_path, ratio_train, 'disjoint', n_clients)
    split_disjoint_subgraphs(n_clients, data, dataset, data_path)


def generate_overlapping_data(dataset, n_comms, n_client_per_comm, data_path, ratio_train):
    data = split_train(get_data(dataset, data_path), dataset, data_path, ratio_train, 'overlapping', n_comms*n_client_per_comm)
    split_overlapping_subgraphs(n_comms, n_client_per_comm, data, dataset, data_path)


def main():
    data_path = '/home/guozhuoning/projects/FGL/data/datasets/'
    ratio_train = 0.2
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args = sys.argv
    mode = args[1]
    dataset = args[2]
    print(mode, dataset)
    if mode == 'disjoint':
        clients = [5, 10, 20]
        for n_clients in clients:
            generate_disjoint_data(dataset=dataset, n_clients=n_clients, data_path=data_path, ratio_train=ratio_train)
    elif mode == 'overlap':
        comms = [2, 6, 10]
        n_client_per_comm = 5
        for n_comms in comms:
            generate_overlapping_data(dataset=dataset, n_comms=n_comms, n_client_per_comm=n_client_per_comm, data_path=data_path, ratio_train=ratio_train)
    else:
        raise Exception(f'{mode} is not disjoint or overlap.')


if __name__ == '__main__':
    main()
