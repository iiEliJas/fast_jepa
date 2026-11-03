"""Benchmark a synthetic LeJEPA training step.

The model and loss are always this repository's implementations; ``--sigreg``
selects which SIGReg module the loss uses (ours or the pinned official one).
Synthetic tensors isolate compute from the data pipeline, as protocol section
5.7 asks for ("report with and without data loading").

The step is timed in three phases so the breakdown is explicit:

* ``forward``: encoder + projector + loss, no backward
* ``forward_backward``: the same plus ``loss.backward()``
* ``full_step``: the same plus ``optimizer.step()`` and ``zero_grad``

Backward time is ``forward_backward - forward``; optimizer time is
``full_step - forward_backward``.  The phases are timed in separate loops so
synchronization does not contaminate the other numbers.
"""

import argparse
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness
import torch

PRECISIONS = {
    "fp32": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def load_components(sigreg_impl, official_root):
    model = harness.load_module_from_path(
        "fj_model_bench", os.path.join(harness.PYSRC, "lejepa", "model.py")
    )
    loss = harness.load_module_from_path(
        "fj_loss_bench", os.path.join(harness.PYSRC, "lejepa", "loss.py")
    )
    if sigreg_impl == "fastjepa":
        sigreg = harness.load_ours_sigreg()
        sig = sigreg.SlicedEppsPulley(16, num_slices=256)
    else:
        slicing, epps = harness.load_official_sigreg(official_root)
        sig = slicing(epps(), num_slices=256)
    return model, loss, sig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigreg", choices=["fastjepa", "official", "both"], default="both")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--V", type=int, default=2)
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--proj-hidden", type=int, default=128)
    parser.add_argument("--proj-dim", type=int, default=16)
    parser.add_argument("--precision", choices=sorted(PRECISIONS), default="fp32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--official-root", default=harness.DEFAULT_OFFICIAL_ROOT)
    parser.add_argument(
        "--out",
        default=os.path.join(harness.REPO_ROOT, "bench", "results", "train.jsonl"),
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)

    if args.quick:
        args.warmup = min(args.warmup, 3)
        args.iters = min(args.iters, 10)

    device = harness.resolve_device(args.device)
    impls = ["fastjepa", "official"] if args.sigreg == "both" else [args.sigreg]
    if args.precision != "fp32" and "official" in impls:
        if args.sigreg == "official":
            parser.error(
                "the official LeJEPA SIGReg is FP32-only in this harness; "
                "use --precision fp32 or --sigreg fastjepa"
            )
        print("official SIGReg is FP32-only; skipping it for --precision " + args.precision)
        impls.remove("official")
    config = {
        "V": args.V,
        "B": args.B,
        "img_size": args.img_size,
        "patch_size": args.patch_size,
        "embed_dim": args.embed_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "proj_hidden": args.proj_hidden,
        "proj_dim": args.proj_dim,
        "precision": args.precision,
        "warmup": args.warmup,
        "iters": args.iters,
        "data": "synthetic",
    }
    autocast_dtype = PRECISIONS[args.precision]

    records = []
    for impl in impls:
        torch.manual_seed(0)
        model_mod, loss_mod, sig = load_components(impl, args.official_root)
        enc = model_mod.LeJEPAEncoder(
            img_size=args.img_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            drop_path_rate=0.0,
            dynamic_img_size=True,
        ).to(device)
        proj = model_mod.Projector(512, args.proj_hidden, args.proj_dim).to(device)
        sig = sig.to(device)
        opt = torch.optim.AdamW(list(enc.parameters()) + list(proj.parameters()), lr=1e-3)
        images = torch.randn(args.B, 3, args.img_size, args.img_size, device=device)

        def autocast():
            if autocast_dtype is not None and device.type == "cuda":
                return torch.autocast(device_type="cuda", dtype=autocast_dtype)
            return contextlib.nullcontext()

        def forward():
            opt.zero_grad(set_to_none=True)
            with autocast():
                emb = enc(images.repeat(args.V, 1, 1, 1))
                z = proj(emb).view(args.V, args.B, -1)
                loss, _, _ = loss_mod.lejepa_loss(z, sig, 0.02)
            return loss

        def forward_backward():
            loss = forward()
            loss.backward()

        def full_step():
            forward_backward()
            opt.step()

        phases = [
            ("forward", forward),
            ("forward_backward", forward_backward),
            ("full_step", full_step),
        ]
        # Ramp clocks and the allocator before the first timed phase.
        for _ in range(10):
            full_step()
        harness.synchronize(device)

        timings = {}
        for name, fn in phases:
            timings[name] = harness.time_call(
                fn, warmup=args.warmup, iters=args.iters, device=device
            )

        peak_mb = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            full_step()
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / 1e6

        images_per_s = (args.V * args.B) / (timings["full_step"]["median_ms"] / 1e3)
        for name, fn in phases:
            timing = timings[name]
            record = harness.base_record(
                "train_step",
                device=device,
                config=config,
                sigreg=impl,
                phase=name,
                images_per_s=images_per_s if name == "full_step" else None,
                peak_mem_mb=peak_mb if name == "full_step" else None,
                **timing,
            )
            records.append(record)
            print(
                f"sigreg={impl:8s} phase={name:16s} "
                f"median={timing['median_ms']:.3f} ms "
                f"[p25={timing['p25_ms']:.3f}, p75={timing['p75_ms']:.3f}]"
            )
        print(
            f"sigreg={impl:8s} full step {images_per_s:.1f} images/s, "
            f"peak {peak_mb if peak_mb is None else round(peak_mb, 1)} MB"
        )

    harness.write_jsonl(args.out, records)
    print(f"\nwrote {len(records)} records to {args.out}")
    return records


if __name__ == "__main__":
    main()
