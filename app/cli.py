"""One-shot CLI, for trying the prompt on a single file without the whole stack.

    python -m app.cli sub.en.srt                 # writes sub.ar.srt next to it
    python -m app.cli sub.en.srt -o out.srt --register gulf
    python -m app.cli movie.mkv --from-video     # extract, then translate
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import sources, srt
from .config import settings
from .providers import build_provider
from .translate import Translator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tarjem", description="Translate a subtitle file to Arabic.")
    parser.add_argument("input", type=Path, help="an .srt file, or a video with --from-video")
    parser.add_argument("-o", "--output", type=Path, help="where to write (default: alongside input)")
    parser.add_argument("--from-video", action="store_true",
                        help="treat input as a video and pull the source subtitle out of it")
    parser.add_argument("--register", default=settings.register, help="msa | msa-light | gulf | egyptian | levantine")
    parser.add_argument("--model", default=None, help="override LLM_MODEL")
    parser.add_argument("--batch-size", type=int, default=settings.batch_size)
    parser.add_argument("--limit", type=int, default=0, help="only translate the first N cues (for a quick taste)")
    parser.add_argument("--no-glossary", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )

    settings.register = args.register
    settings.batch_size = args.batch_size
    settings.glossary_enabled = not args.no_glossary
    if args.model:
        settings.model = args.model
        settings.openai_model = args.model

    if args.from_video:
        found = sources.find_source(args.input, settings)
        if not found:
            print(f"no usable source subtitle in {args.input}", file=sys.stderr)
            return 2
        print(f"source: {found.origin} - {found.detail}", file=sys.stderr)
        text = found.text
    else:
        text = srt.decode(args.input.read_bytes())

    cues = srt.parse(text)
    if not cues:
        print("no cues parsed - is that really a subtitle file?", file=sys.stderr)
        return 2
    if args.limit:
        cues = cues[:args.limit]
    print(f"{len(cues)} cues", file=sys.stderr)

    provider = build_provider(settings)
    translator = Translator(provider, settings)
    title = args.input.stem

    glossary = translator.build_glossary(cues, title)
    if glossary:
        print(f"glossary: {len(glossary.terms)} terms", file=sys.stderr)

    def progress(fraction: float, stage: str) -> None:
        print(f"  {stage} ({fraction:.0%})", file=sys.stderr)

    translated, stats = translator.translate(cues, title, glossary, progress)
    provider.close()

    out = args.output or args.input.with_suffix("").with_name(args.input.stem + settings.output_suffix)
    out.write_text(srt.dumps(translated), encoding="utf-8")
    print(f"\nwrote {out}", file=sys.stderr)
    print(f"stats: {stats.as_dict()}", file=sys.stderr)
    print(f"usage: {provider.usage.as_dict()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
