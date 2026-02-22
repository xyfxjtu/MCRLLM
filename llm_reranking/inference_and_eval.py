import sys
sys.path.append(".")
from reranking_strategy import WeightedFusion, MultiLevel, SelfGeneratedExample
import argparse


datasets = ["WN18RR_IMG", "FB15K_237_IMG"]
rerank = ["MW", "ML", "SPN"]
mode = ["Full", "Sampling"]

parser = argparse.ArgumentParser(
    description="Relational learning contraption"
)

parser.add_argument(
    '--dataset', choices=datasets,
    default='WN18RR_IMG',
    help="Dataset in {}".format(datasets)
)

parser.add_argument(
    '--rerank', choices=rerank,
    default="MW",
    help="rerank strategy."
)

parser.add_argument(
    '--mode', choices=mode,
    default="Full",
    help="inference mode"
)
parser.add_argument(
    '--sampling_rate', default=0.01, type=float,
    help="sampling rate."
)
parser.add_argument(
    '--k', default=20, type=int,
    help="number of candidate entities"
)

args = parser.parse_args()
print(args)

model = None
if args.rerank == "MW":
    model = WeightedFusion(args.dataset, args.k, args.mode, args.sampling_rate)
elif args.rerank == "ML":
    model = MultiLevel(args.dataset, args.k, args.mode, args.sampling_rate)
else:
    model = SelfGeneratedExample(args.dataset, args.k, args.mode, args.sampling_rate)

model.eval()
