"""Command line interface. Run from the repository root."""

import argparse
from pathlib import Path

from .audit import audit
from .evaluate import import_rollouts, offline, simulate
from .experiments import make_plan, train
from .io import read_json, write_json
from .report import report, write_report


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    specs = {
        "audit": ("Scan LeRobot v3 episodes", ("dataset", "protocol", "episodes", "out")),
        "plan": ("Freeze train subsets and the evaluation protocol", ("audit", "protocol", "recipe", "out")),
        "train": ("Run the frozen LeRobot training jobs", ("dataset", "plan", "out")),
        "offline": (
            "Evaluate predicted chunks on a frozen reference split",
            ("dataset", "plan", "protocol", "run", "out"),
        ),
        "sim": ("Run native lerobot-eval on frozen simulation cases", ("plan", "run", "out")),
        "import-rollouts": ("Import independently verified outcomes", ("plan", "run", "records", "out")),
        "report": ("Generate the quality and effectiveness report", ("audit", "protocol", "plan", "out")),
    }
    sub = {}
    for name, (help_text, paths) in specs.items():
        command = commands.add_parser(name, help=help_text)
        sub[name] = command
        for field in paths:
            command.add_argument(f"--{field}", type=Path, required=True)
    sub["audit"].add_argument("--raw-timing", type=Path)
    sub["plan"].add_argument("--cases", type=Path)
    sub["train"].add_argument("--variant", action="append")
    sub["train"].add_argument("--seed", type=int, action="append")
    sub["offline"].add_argument("--split", choices=("dev", "test"), default="dev")
    sub["offline"].add_argument("--device", default="cpu")
    sub["offline"].add_argument("--stride", type=int, default=1)
    sub["offline"].add_argument("--draws", type=int, default=1)
    sub["report"].add_argument("--offline", type=Path, nargs="*", default=[])
    sub["report"].add_argument("--rollouts", type=Path, nargs="*", default=[])
    return parser, parser.parse_args()


def main():
    parser, args = arguments()
    try:
        if args.command == "audit":
            result = audit(args.dataset, read_json(args.protocol), args.episodes, args.raw_timing)
            write_json(args.out, result)
        elif args.command == "plan":
            result = make_plan(
                read_json(args.audit),
                read_json(args.protocol),
                read_json(args.recipe),
                read_json(args.cases) if args.cases else None,
            )
            write_json(args.out, result)
        elif args.command == "train":
            paths = train(args.dataset, read_json(args.plan), args.out, args.variant, args.seed)
            for path in paths:
                print(path)
            return
        elif args.command == "offline":
            result = offline(
                args.dataset,
                read_json(args.plan),
                read_json(args.protocol),
                read_json(args.run),
                split=args.split,
                device=args.device,
                stride=args.stride,
                draws=args.draws,
            )
            write_json(args.out, result)
        elif args.command == "sim":
            simulate(read_json(args.plan), read_json(args.run), args.out)
        elif args.command == "import-rollouts":
            result = import_rollouts(read_json(args.plan), read_json(args.run), args.records)
            write_json(args.out, result)
        elif args.command == "report":
            source = read_json(args.audit)
            result = report(
                source,
                read_json(args.protocol),
                read_json(args.plan),
                [read_json(p) for p in args.offline],
                [read_json(p) for p in args.rollouts],
            )
            write_report(args.out, result, source)
    except (ValueError, KeyError, OSError, ImportError, TypeError) as exc:
        parser.exit(2, f"vla_eval: {exc}\n")
    print(args.out)


if __name__ == "__main__":
    main()
