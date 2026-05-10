#!/usr/bin/env python3
"""各フレームで text prompt 検出をやり直す benchmark。"""

from __future__ import annotations

import argparse

from video_text_prompt_benchmark_utils import (
    Timer,
    add_common_video_args,
    build_video_model,
    finalize_summary,
    get_end_frame,
    init_video_state,
    make_frame_record,
    print_summary,
    sync_cuda,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark per-frame SAM3 text-prompt detection on a video/image sequence"
    )
    add_common_video_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = build_video_model(args)
    state = init_video_state(model, args)
    end_frame = get_end_frame(state, args)

    frame_records = []
    with Timer() as total_timer:
        for frame_idx in range(args.start_frame, end_frame + 1):
            with Timer() as frame_timer:
                _, output = model.add_prompt(
                    state,
                    frame_idx=frame_idx,
                    text_str=args.prompt,
                )
                sync_cuda(args.device)
            frame_records.append(
                make_frame_record(
                    frame_idx=frame_idx,
                    elapsed_sec=frame_timer.elapsed,
                    output=output,
                    phase="detect",
                )
            )

    summary = finalize_summary(
        mode="per_frame_text_prompt",
        prompt=args.prompt,
        input_path=args.input,
        frame_records=frame_records,
        total_elapsed_sec=total_timer.elapsed,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
