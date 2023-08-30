from datetime import datetime
import os

from config.parser import Parser
from module.others.multiprocs import ParentProcess


def set_config(args):

    if args.dataset == 'Cora':
        args.n_feat = 1433
        args.n_clss = 7

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial = f'{args.dataset}_{args.mode}/clients_{args.n_clients}/{now}_{args.model}'
    args.data_path = f'{args.base_path}/data/datasets'
    args.checkpt_path = f'{args.base_path}/checkpoint/{trial}'
    args.log_path = f'{args.base_path}/log/{trial}'

    return args


if __name__ == '__main__':
    args = set_config(Parser().parse())
    if args.federated == 'fedavg':
        from module.federated.fedavg.server import Server
        from module.federated.fedavg.client import Client
    elif args.federated == 'fedpub':
        from module.federated.fedpub.server import Server
        from module.federated.fedpub.client import Client
    else:
        print('incorrect model was given: {}'.format(args.model))
        os._exit(0)
    pp = ParentProcess(args, Server, Client)
    pp.start()
