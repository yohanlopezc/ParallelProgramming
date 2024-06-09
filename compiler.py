import argparse
from Keras import horovodize as hvd_tfkeras
from Torch import horovodize as hvd_torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fw', '--framework',  dest='fw', type=str, help="FrameWork used by the original script", choices=['tf1', 'tf2', 'torch'], required=True)
    parser.add_argument('-fp', '--filepath',  dest='fp', type=str, help="Path to the original scripts", required=True)
    parser.add_argument('-v', '--verbose',  dest='verbose', type=bool, help="Verbose", required=False, default=False)
    args = parser.parse_args()
    if args.verbose:
        print(f"Arguments:\n  --framework: {args.fw}\n  --filepath: {args.fp}")
    if args.fw == 'torch':
        hvd_torch(args.fp, args.verbose)
    elif args.fw in ['tf1', 'tf2']:
        hvd_tfkeras(args.fw, args.fp, args.verbose)
