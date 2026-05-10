#!/usr/bin/env python3
"""初回だけ text prompt を与え、その後は tracking のみで流す benchmark。"""

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
        description="Benchmark initial text prompt detection followed by SAM3 tracking"
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
        with Timer() as prompt_timer:
            prompt_frame_idx, prompt_output = model.add_prompt(
                state,
                frame_idx=args.start_frame,
                text_str=args.prompt,
            )
            sync_cuda(args.device)
        frame_records.append(
            make_frame_record(
                frame_idx=prompt_frame_idx,
                elapsed_sec=prompt_timer.elapsed,
                output=prompt_output,
                phase="detect",
            )
        )

        track_span = end_frame - args.start_frame
        if track_span > 0:
            for frame_idx, output in model.propagate_in_video(
                state,
                start_frame_idx=args.start_frame + 1,
                max_frame_num_to_track=track_span - 1,
                reverse=False,
            ):
                sync_cuda(args.device)
                frame_records.append(
                    make_frame_record(
                        frame_idx=frame_idx,
                        elapsed_sec=0.0,
                        output=output,
                        phase="track",
                    )
                )

    summary = finalize_summary(
        mode="text_prompt_then_track",
        prompt=args.prompt,
        input_path=args.input,
        frame_records=frame_records,
        total_elapsed_sec=total_timer.elapsed,
        extra={"refresh_count": 0},
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
