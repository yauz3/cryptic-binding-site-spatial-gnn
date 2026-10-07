#!/usr/bin/env python3
import argparse
from cryptogrid.common import load_folds, split_ids
ap=argparse.ArgumentParser(); ap.add_argument("folds"); ap.add_argument("--split",default="test"); args=ap.parse_args()
tr,te,tf=split_ids(load_folds(args.folds))
ids=te if args.split=="test" else (tf.get(args.split,tr) if args.split!="train" else tr)
print("\n".join(ids))
