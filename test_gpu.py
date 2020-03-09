import torch
import argparse


print("GPU is available: ", torch.cuda.is_available())

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    print("Using GPU")
else:
    args.device = torch.device('cpu')
    print("Using CPU")

with torch.cuda.device(args.device):
    print("Inside device:", args.device)  # On device 1
