"""Benchmark SIGReg forward and forward+backward latency.

Compares this repository's ``SlicedEppsPulley`` with the pinned official
``SlicingUnivariateTest`` over the N grid from protocol section 5.7.  Results
go to one JSONL file per run, with device/driver/commit/config attached to
every record.

Examples
--------
Full grid, both implementations, 2D and 4-view inputs::

    .venv/bin/python bench/bench_sigreg.py --out bench/results/sigreg.jsonl

Quick check::

    .venv/bin/python bench/bench_sigreg.py --quick --impl fastjepa --views 1
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness
import torch

DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}
FULL_N = [256, 512, 1024, 2048, 4096, 8192]
QUICK_N = [256, 1024]


def build_module(impl, k, num_slices, t_max, n_knots, device, official_root):
    if impl == "fastjepa":
        sigreg = harness.load_ours_sigreg()
        module = sigreg.SlicedEppsPulley(k, num_slices=num_slices, t_max=t_max, n_knots=n_knots)
    else:
        slicing, epps = harness.load_official_sigreg(official_root)
        module = slicing(epps(t_max=t_max, n_points=n_knots), num_slices=num_slices)
        # The official implementation samples its projection matrix in FP32 and
        # does not support fp16/bf16 inputs in this harness.
    return module.to(device)


def make_forward(module, x):
    return lambda: module(x)


def make_backward(module, x):
    def run():
        module(x).backward()
        x.grad = None

    return run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=["fastjepa", "official", "both"], default="both")
    parser.add_argument("--mode", choices=["forward", "backward", "both"], default="both")
    parser.add_argument("--views", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--N", type=int, nargs="+", default=None)
    parser.add_argument("--K", type=int, default=16)
    parser.add_argument("--S", type=int, default=256)
    parser.add_argument("--t-max", type=float, default=3.0)
    parser.add_argument("--n-knots", type=int, default=17)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--official-root", default=harness.DEFAULT_OFFICIAL_ROOT)
    parser.add_argument(
        "--out",
        default=os.path.join(harness.REPO_ROOT, "bench", "results", "sigreg.jsonl"),
    )
    parser.add_argument("--quick", action="store_true", help="small N grid and fewer iterations")
    args = parser.parse_args(argv)

    if args.quick:
        args.N = args.N or QUICK_N
        args.warmup = min(args.warmup, 5)
        args.iters = min(args.iters, 20)
    args.N = args.N or FULL_N

    device = harness.resolve_device(args.device)
    dtype = DTYPES[args.dtype]
    impls = ["fastjepa", "official"] if args.impl == "both" else [args.impl]
    if args.dtype != "fp32" and "official" in impls:
        if args.impl == "official":
            parser.error(
                "the official LeJEPA SIGReg is FP32-only in this harness; "
                "use --dtype fp32 or --impl fastjepa"
            )
        print("official SIGReg is FP32-only; skipping it for --dtype " + args.dtype)
        impls.remove("official")
    modes = ["forward", "backward"] if args.mode == "both" else [args.mode]

    # Load both modules before timing so import and one-time buffer setup are
    # not part of any sample.
    modules = {}
    for impl in impls:
        modules[impl] = build_module(
            impl,
            args.K,
            args.S,
            args.t_max,
            args.n_knots,
            device,
            args.official_root,
        )

    config = {
        "K": args.K,
        "S": args.S,
        "t_max": args.t_max,
        "n_knots": args.n_knots,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iters": args.iters,
    }
    # Ramp GPU clocks before the first measured configuration; otherwise the
    # smallest N can look slower than the next one.
    if device.type == "cuda":
        ramp = torch.randn(4096, args.K, device=device, dtype=dtype)
        for _ in range(50):
            for module in modules.values():
                module(ramp)
        harness.synchronize(device)

    records = []
    for views in args.views:
        for n in args.N:
            shape = (n, args.K) if views == 1 else (views, n, args.K)
            x = torch.randn(*shape, device=device, dtype=dtype)
            for impl in impls:
                for mode in modes:
                    if mode == "backward":
                        x.requires_grad_(True)
                        fn = make_backward(modules[impl], x)
                    else:
                        x.requires_grad_(False)
                        fn = make_forward(modules[impl], x)
                    timing = harness.time_call(
                        fn, warmup=args.warmup, iters=args.iters, device=device
                    )
                    record = harness.base_record(
                        "sigreg",
                        device=device,
                        config=config,
                        impl=impl,
                        mode=mode,
                        views=views,
                        shape=list(shape),
                        N=n,
                        K=args.K,
                        S=args.S,
                        dtype=args.dtype,
                        module_dtype=("fp32" if impl == "official" else "fp32_statistics"),
                        **timing,
                    )
                    records.append(record)
                    print(
                        f"{impl:8s} {mode:8s} views={views} N={n:5d} "
                        f"median={timing['median_ms']:.4f} ms "
                        f"[p25={timing['p25_ms']:.4f}, p75={timing['p75_ms']:.4f}]"
                    )

    harness.write_jsonl(args.out, records)
    print(f"\nwrote {len(records)} records to {args.out}")
    return records


if __name__ == "__main__":
    main()
