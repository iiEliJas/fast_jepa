"""One-command benchmark suite for the Python LeJEPA baseline.

Runs the SIGReg microbenchmark and the synthetic training-step benchmark, then
prints the collected numbers as markdown tables.  Raw JSONL stays in
``bench/results/`` and is what ``docs/experiments/001_sigreg_benchmark.md``
reports.

Examples
--------
Full run (RTX 4060 takes roughly two minutes)::

    .venv/bin/python bench/run_benchmarks.py

Quick smoke test::

    .venv/bin/python bench/run_benchmarks.py --quick
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
FULL_N = [256, 512, 1024, 2048, 4096, 8192]
VIEW_N = [256, 512, 1024, 2048]
QUICK_N = [256, 1024]
QUICK_VIEW_N = [256, 1024]


def run(command):
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=harness.REPO_ROOT)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-sigreg", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--impl",
        choices=["both", "fastjepa", "official"],
        default="both",
        help="which SIGReg implementations to run",
    )
    parser.add_argument("--official-root", default=harness.DEFAULT_OFFICIAL_ROOT)
    parser.add_argument(
        "--results-dir",
        default=os.path.join(harness.REPO_ROOT, "bench", "results"),
    )
    args = parser.parse_args(argv)

    impl = args.impl
    if impl in ("both", "official") and not harness.has_official(args.official_root):
        if impl == "official":
            parser.error(f"official lejepa checkout not found at {args.official_root}")
        print(
            f"official checkout not found at {args.official_root}; "
            "running the fastjepa implementation only"
        )
        impl = "fastjepa"

    python = sys.executable
    sigreg_out = os.path.join(args.results_dir, "sigreg.jsonl")
    train_out = os.path.join(args.results_dir, "train.jsonl")
    common = ["--official-root", args.official_root]
    if args.quick:
        common.append("--quick")
    n_grid = QUICK_N if args.quick else FULL_N
    view_grid = QUICK_VIEW_N if args.quick else VIEW_N

    if not args.skip_sigreg:
        run(
            [
                python,
                os.path.join(BENCH_DIR, "bench_sigreg.py"),
                "--impl",
                impl,
                "--mode",
                "both",
                "--views",
                "1",
                "--N",
                *[str(n) for n in n_grid],
                "--out",
                sigreg_out,
                *common,
            ]
        )
        run(
            [
                python,
                os.path.join(BENCH_DIR, "bench_sigreg.py"),
                "--impl",
                impl,
                "--mode",
                "forward",
                "--views",
                "4",
                "--N",
                *[str(n) for n in view_grid],
                "--out",
                os.path.join(args.results_dir, "sigreg_views.jsonl"),
                *common,
            ]
        )

    if not args.skip_train:
        run(
            [
                python,
                os.path.join(BENCH_DIR, "bench_train.py"),
                "--sigreg",
                impl,
                "--out",
                train_out,
                *common,
            ]
        )

    print("\n## SIGReg forward/backward (2D, K=16, S=256)\n")
    if os.path.exists(sigreg_out):
        records = harness.read_jsonl(sigreg_out)
        records = [r for r in records if r["dtype"] == "fp32"]
        print(
            harness.format_table(
                records,
                [
                    ("impl", "impl"),
                    ("mode", "mode"),
                    ("N", "N"),
                    ("median_ms", "median ms"),
                    ("p25_ms", "p25 ms"),
                    ("p75_ms", "p75 ms"),
                ],
            )
        )

    print("\n## SIGReg forward with 4 views (K=16, S=256)\n")
    views_out = os.path.join(args.results_dir, "sigreg_views.jsonl")
    if os.path.exists(views_out):
        records = harness.read_jsonl(views_out)
        records = [r for r in records if r["dtype"] == "fp32"]
        print(
            harness.format_table(
                records,
                [
                    ("impl", "impl"),
                    ("N", "N"),
                    ("median_ms", "median ms"),
                    ("p25_ms", "p25 ms"),
                    ("p75_ms", "p75 ms"),
                ],
            )
        )

    print("\n## Synthetic training step (V=2, B=8, 32x32, 2-layer ViT)\n")
    if os.path.exists(train_out):
        records = harness.read_jsonl(train_out)
        print(
            harness.format_table(
                records,
                [
                    ("sigreg", "SIGReg"),
                    ("phase", "phase"),
                    ("median_ms", "median ms"),
                    ("p25_ms", "p25 ms"),
                    ("p75_ms", "p75 ms"),
                    ("images_per_s", "images/s"),
                    ("peak_mem_mb", "peak MB"),
                ],
            )
        )

    print(f"\nraw results: {args.results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
