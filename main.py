import argparse
from lib.args import build_parser, params_from_args
from lib.runner import run_and_save

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    params = params_from_args(args)
    run_and_save(
        params,
        out_dir=args.out_dir,
        save_test_batch=args.save_test_batch,
        save_cnn_features=args.save_cnn_features,
        tag=args.tag,
    )

