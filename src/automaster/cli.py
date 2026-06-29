"""Command-line interface: master a file with a trained editor model.

    automaster master in.wav -o out.wav --editor boris [--no-limiter]
    automaster analyze data/raw -o data/reports
"""
from __future__ import annotations

import argparse
from pathlib import Path

from automaster import metrics


def _master(args):
    from automaster import render, io
    from automaster.model import EditorModel

    model = EditorModel.load(args.model or _default_model(args.editor))
    x, sr = io.load_audio(args.input)
    before = metrics.measure(x, sr)
    y = render.render(x, sr, model, apply_limiter=not args.no_limiter,
                      ceiling_db=args.ceiling)
    io.save_audio(args.output, y, sr)
    after = metrics.measure(y, sr)
    print(f"in : {before['lufs']:.2f} LUFS  {before['tp_db']:.2f} dBTP")
    print(f"out: {after['lufs']:.2f} LUFS  {after['tp_db']:.2f} dBTP  -> {args.output}")


def _analyze(args):
    from automaster import analyze
    df = analyze.analyze_corpus(args.root, args.out)
    print(df.to_string(index=False))


def _default_model(editor):
    p = Path(__file__).resolve().parent.parent.parent / "models" / f"{editor}.json"
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(prog="automaster")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("master", help="master a file with an editor model")
    m.add_argument("input")
    m.add_argument("-o", "--output", required=True)
    m.add_argument("--editor", default="boris")
    m.add_argument("--model", help="path to a model JSON (overrides --editor)")
    m.add_argument("--no-limiter", action="store_true",
                   help="replicate mode: allow peaks over the ceiling")
    m.add_argument("--ceiling", type=float, default=-1.0)
    m.set_defaults(func=_master)

    a = sub.add_parser("analyze", help="compute deltas + scatter for a corpus")
    a.add_argument("root")
    a.add_argument("-o", "--out", default="data/reports")
    a.set_defaults(func=_analyze)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
