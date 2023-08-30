import time
import numpy as np

from scipy.spatial.distance import cosine

from module.utils import *
from module.model import *
from module.federated.base import ServerModule


class Server(ServerModule):
    def __init__(self, args, sd, gpu_server):
        super(Server, self).__init__(args, sd, gpu_server)
        self.model = MaskedGCN(self.args.n_feat, self.args.n_dims, self.args.n_clss, self.args.l1, self.args).cuda(self.gpu_id)
        self.sd['proxy'] = self.get_proxy_data(args.n_feat)
        self.update_lists = []
        self.sim_matrices = []

    def get_proxy_data(self, n_feat):
        import networkx as nx

        num_graphs, num_nodes = self.args.n_proxy, 100
        data = self.from_networkx(nx.random_partition_graph([num_nodes] * num_graphs, p_in=0.1, p_out=0, seed=self.args.seed))
        data.x = torch.normal(mean=0, std=1, size=(num_nodes * num_graphs, n_feat))
        return data

    def on_round_begin(self, curr_rnd):
        self.round_begin = time.time()
        self.curr_rnd = curr_rnd
        self.sd['global'] = self.get_weights()

    def on_round_complete(self, updated):
        self.update(updated)
        self.save_state()

    def update(self, updated):
        st = time.time()
        local_weights = []
        local_functional_embeddings = []
        local_train_sizes = []
        for c_id in updated:
            local_weights.append(self.sd[c_id]['model'].copy())
            local_functional_embeddings.append(self.sd[c_id]['functional_embedding'])
            local_train_sizes.append(self.sd[c_id]['train_size'])
            del self.sd[c_id]
        self.logger.print(f'all clients have been uploaded ({time.time()-st:.2f}s)')

        n_connected = round(self.args.n_clients*self.args.frac)
        assert n_connected == len(local_functional_embeddings)
        sim_matrix = np.empty(shape=(n_connected, n_connected))
        for i in range(n_connected):
            for j in range(n_connected):
                sim_matrix[i, j] = 1 - cosine(local_functional_embeddings[i], local_functional_embeddings[j])

        if self.args.agg_norm == 'exp':
            sim_matrix = np.exp(self.args.norm_scale * sim_matrix)

        row_sums = sim_matrix.sum(axis=1)
        sim_matrix = sim_matrix / row_sums[:, np.newaxis]

        st = time.time()
        ratio = (np.array(local_train_sizes)/np.sum(local_train_sizes)).tolist()
        self.set_weights(self.model, self.aggregate(local_weights, ratio))
        self.logger.print(f'global model has been updated ({time.time()-st:.2f}s)')

        st = time.time()
        for i, c_id in enumerate(updated):
            aggr_local_model_weights = self.aggregate(local_weights, sim_matrix[i, :])
            if f'personalized_{c_id}' in self.sd:
                del self.sd[f'personalized_{c_id}']
            self.sd[f'personalized_{c_id}'] = {'model': aggr_local_model_weights}
        self.update_lists.append(updated)
        self.sim_matrices.append(sim_matrix)
        self.logger.print(f'local model has been updated ({time.time()-st:.2f}s)')

    def set_weights(self, model, state_dict):
        set_state_dict(model, state_dict, self.gpu_id)

    def get_weights(self):
        return {
            'model': get_state_dict(self.model),
        }

    def save_state(self):
        torch_save(self.args.checkpt_path, 'server_state.pt', {
            'model': get_state_dict(self.model),
            'sim_matrices': self.sim_matrices,
            'update_lists': self.update_lists
        })

    def from_networkx(self, G, group_node_attrs=None, group_edge_attrs=None):
        import networkx as nx
        from torch_geometric.data import Data
        from collections import defaultdict

        G = G.to_directed() if not nx.is_directed(G) else G

        mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
        edge_index = torch.empty((2, G.number_of_edges()), dtype=torch.long)
        for i, (src, dst) in enumerate(G.edges()):
            edge_index[0, i] = mapping[src]
            edge_index[1, i] = mapping[dst]

        data = defaultdict(list)

        if G.number_of_nodes() > 0:
            node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
        else:
            node_attrs = {}

        if G.number_of_edges() > 0:
            edge_attrs = list(next(iter(G.edges(data=True)))[-1].keys())
        else:
            edge_attrs = {}

        for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
            if set(feat_dict.keys()) != set(node_attrs):
                raise ValueError('Not all nodes contain the same attributes')
            for key, value in feat_dict.items():
                data[str(key)].append(value)

        for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
            if set(feat_dict.keys()) != set(edge_attrs):
                raise ValueError('Not all edges contain the same attributes')
            for key, value in feat_dict.items():
                key = f'edge_{key}' if key in node_attrs else key
                data[str(key)].append(value)

        for key, value in G.graph.items():
            if key == 'node_default' or key == 'edge_default':
                continue  # Do not load default attributes.
            key = f'graph_{key}' if key in node_attrs else key
            data[str(key)] = value

        for key, value in data.items():
            if isinstance(value, (tuple, list)) and isinstance(value[0], torch.Tensor):
                data[key] = torch.stack(value, dim=0)
            else:
                try:
                    data[key] = torch.tensor(value)
                except (ValueError, TypeError, RuntimeError):
                    pass

        data['edge_index'] = edge_index.view(2, -1)
        data = Data.from_dict(data)

        if group_node_attrs is all:
            group_node_attrs = list(node_attrs)
        if group_node_attrs is not None:
            xs = []
            for key in group_node_attrs:
                x = data[key]
                x = x.view(-1, 1) if x.dim() <= 1 else x
                xs.append(x)
                del data[key]
            data.x = torch.cat(xs, dim=-1)

        if group_edge_attrs is all:
            group_edge_attrs = list(edge_attrs)
        if group_edge_attrs is not None:
            xs = []
            for key in group_edge_attrs:
                key = f'edge_{key}' if key in node_attrs else key
                x = data[key]
                x = x.view(-1, 1) if x.dim() <= 1 else x
                xs.append(x)
                del data[key]
            data.edge_attr = torch.cat(xs, dim=-1)

        if data.x is None and data.pos is None:
            data.num_nodes = G.number_of_nodes()

        return data
