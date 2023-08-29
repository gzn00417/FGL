import torch
import os


class DataLoader:
    def __init__(self, args):
        self.args = args
        self.n_workers = 1
        self.client_id = None

        from torch_geometric.loader import DataLoader
        self.DataLoader = DataLoader

    def switch(self, client_id):
        if self.client_id == client_id:
            return
        self.client_id = client_id
        self.partition = [torch.load(
            os.path.join(self.args.data_path, f'{self.args.dataset}_{self.args.mode}/{self.args.n_clients}/partition_{client_id}.pt'),
            map_location=torch.device('cpu')
        )['client_data']]
        self.pa_loader = self.DataLoader(dataset=self.partition, batch_size=1, shuffle=False, num_workers=self.n_workers, pin_memory=False)
