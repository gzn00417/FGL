import argparse


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()

    def set_arguments(self):
        self.parser.add_argument('--gpu', type=str, default='0')
        self.parser.add_argument('--seed', type=int, default=1234)

        self.parser.add_argument('--federated', type=str, default='fedavg')
        self.parser.add_argument('--model', type=str, default='GCN', choices=['GCN'])
        self.parser.add_argument('--dataset', type=str, default='Cora')
        self.parser.add_argument('--mode', type=str, default='disjoint', choices=['disjoint', 'overlapping'])
        self.parser.add_argument('--base-path', type=str, default='./')

        self.parser.add_argument('--n-workers', type=int, default=2)
        self.parser.add_argument('--n-clients', type=int, default=10)
        self.parser.add_argument('--n-rnds', type=int, default=100)
        self.parser.add_argument('--n-eps', type=int, default=1)
        self.parser.add_argument('--frac', type=float, default=1.0)
        self.parser.add_argument('--n-dims', type=int, default=128)
        self.parser.add_argument('--lr', type=float, default=0.1)
        self.parser.add_argument('--weight_decay', type=float, default=1e-6)
        self.parser.add_argument('--l1', type=float, default=1e-3)

    def parse(self):
        args, unparsed = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args