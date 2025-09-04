import argparse
from dataclasses import fields, MISSING
from .config import ExperimentParams


def parse_float_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",")] if s else []


def dataclass_defaults(cls):
    d = {}
    for f in fields(cls):
        if f.init is False:
            continue
        if f.default is not MISSING:
            d[f.name] = f.default
        elif f.default_factory is not MISSING:  # type: ignore[attr-defined]
            d[f.name] = f.default_factory()  # type: ignore[misc]
    return d


def build_parser() -> argparse.ArgumentParser:
    dc_defaults = dataclass_defaults(ExperimentParams)

    parser = argparse.ArgumentParser(description="Run causal_vision experiment")
    # Core task setup
    parser.add_argument("--dims", type=int, default=dc_defaults["dims"])
    parser.add_argument("--input-type", choices=["spatial", "feature"], default=dc_defaults["input_type"])
    parser.add_argument("--input-size", type=int, default=dc_defaults["input_size"])
    # Represent list defaults as comma-separated strings
    parser.add_argument("--p", type=str, default=",".join(map(str, dc_defaults["p"])), help="Comma-separated list of p values, e.g. '0.4' or '0.4,0.6'")
    parser.add_argument("--q", type=float, default=dc_defaults["q"])
    parser.add_argument("--kappas", type=str, default=",".join(map(str, dc_defaults["kappas"])), help="Comma-separated list of kappas, e.g. '7.0' or '7.0,8.0'")

    # Model architecture
    parser.add_argument("--hidden-size", type=int, default=dc_defaults["hidden_size"])
    parser.add_argument("--output-type", choices=["angle", "feature", "angle_color"], default=dc_defaults["output_type"])

    # Stimulus tuning
    parser.add_argument("--tuning-concentration", type=float, default=dc_defaults["tuning_concentration"])
    parser.add_argument("--A", type=float, default=dc_defaults["A"])

    # RNN dynamics
    parser.add_argument("--dt", type=float, default=dc_defaults["dt"])
    parser.add_argument("--tau", type=float, default=dc_defaults["tau"])
    parser.add_argument("--eps1", type=float, default=dc_defaults["eps1"])
    parser.add_argument("--C", type=float, default=dc_defaults["C"])

    # Training
    parser.add_argument("--lr", type=float, default=dc_defaults["lr"])
    parser.add_argument("--batch-size", type=int, default=dc_defaults["batch_size"])
    parser.add_argument("--num-batches", type=int, default=dc_defaults["num_batches"])
    parser.add_argument("--interactive", action=argparse.BooleanOptionalAction, default=dc_defaults["interactive"])

    # Testing
    parser.add_argument("--test-batch-size", type=int, default=dc_defaults["test_batch_size"])

    # Output paths and saving toggles (not in dataclass)
    parser.add_argument("--out-dir", type=str, default="output")
    parser.add_argument("--save-test-batch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-cnn-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tag", type=str, default="", help="Optional tag appended to exp dir name")
    return parser


def params_from_args(args) -> ExperimentParams:
    p_list = parse_float_list(args.p)
    kappas_list = parse_float_list(args.kappas)

    return ExperimentParams(
        dims=args.dims,
        input_type=args.input_type,
        input_size=args.input_size,
        p=p_list,
        q=args.q,
        kappas=kappas_list,
        hidden_size=args.hidden_size,
        output_type=args.output_type,
        tuning_concentration=args.tuning_concentration,
        A=args.A,
        dt=args.dt,
        tau=args.tau,
        eps1=args.eps1,
        C=args.C,
        lr=args.lr,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        interactive=args.interactive,
        test_batch_size=args.test_batch_size,
    )
