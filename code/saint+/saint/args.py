import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of multihead attention layer(default: 2)')
parser.add_argument('--num_heads', type=int, default=4,
                    help='number of head in one multihead attention layer(default: 3)')
parser.add_argument('--model_dim', type=int, default=128,
                    help='dimension of embedding size(default: 64)')
parser.add_argument('--max_len', type=int, default=1000,
                    help='maximum index for position encoding(default: 1000)')
parser.add_argument('--num_question', type=int, default=9454,
                    help='number of different question(default: 13523)')
parser.add_argument('--num_test', type=int, default=1537,
                    help='number of different task id(default: 10000)')
parser.add_argument('--seq_len', type=int, default=100,
                    help='sequence length(default: 100)')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout ratio(default: 0.1)')
parser.add_argument('--epochs', type=int, default=250,
                    help='number of epochs(default: 30)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size(default: 512)')
parser.add_argument('--learning_rate', type=float, default=0.05,
                    help='learning rate(default:0.05)')
parser.add_argument('--warmup', type=int, default=4000,
                    help='warmup(default:4000)')